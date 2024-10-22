import os
import math
import json
from tqdm.auto import tqdm
import logging
import argparse
from pathlib import Path
import random
import time
import wandb

import datasets
from datasets import DatasetDict

import transformers
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    DataCollatorForLanguageModeling,
    get_scheduler,
    SchedulerType,
)
from transformers.utils import send_example_telemetry

import torch
from torch.utils.data import DataLoader 
from torch.optim import AdamW

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from huggingface_hub import HfApi

from early_stopping import *

##### track
logger = get_logger(__name__)

##### DEFINE HYPERPARAMETERS
def parse_args(): 
    parser = argparse.ArgumentParser(description="Finetune a BART base model on a PubMed summarization dataset")
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )  
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",    
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default='facebook/bart-base',
        help="Model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--num_train_epochs", 
                        type=int, 
                        default=100, 
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--output_dir", 
                        type=str, 
                        default='output/', 
                        help="Where to store the final model.")
    parser.add_argument("--seed", 
                        type=int, 
                        default=None, 
                        help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", default=True, action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        default=True,
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--hub_token", type=str, default="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", help="The token to use to push to the Model Hub.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # track the example usage --> better allocate resources to maintain them
    send_example_telemetry("bart_base", args)

    ##### Initialize accelerator 
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)   

    # make 1 log on every process with configuration for debugging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )      
    logger.info(accelerator.state, main_process_only=False)

    # if local main process -> more informative logs; subprocess: ERROR only
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    # main_process: in distributed training, it coordinates the overall training execution & interacts with the users
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = 'pretrain_bart_pubmed'
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id
            
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
    accelerator.wait_for_everyone()    

    ##### GET THE DATASET
    # In distributed training, the load_dataset func guarantee that only one local process can concurrently download the dataset 
    raw_datasets = datasets.load_dataset('cppmai/pretrain_pubmed_100k')
    # train_samples = raw_datasets['train'].select(range(50))
    # valid_samples = raw_datasets['validation'].select(range(5))
    # raw_datasets = DatasetDict({
    #     'train': train_samples,
    #     'validation': valid_samples
    # })
    
    ##### Get pretrained model and tokenizer 
    # in distributed training, .from_pretrained methods guarantee that only 1 local process can concurrently down model & vocab
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    # download the pretrained model and fine-tune
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    ##### PREPROCESSING
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    column_names = raw_datasets["train"].column_names
    
    def preprocess_function(examples):
        # add prefix to the inputs
        inputs = [f'Summary: {s} Document: {d}' for s, d in zip(examples['abstract'], examples['article'])]
        # tokenizer documents
        model_inputs = tokenizer(inputs, 
                                 max_length=args.max_source_length, 
                                 truncation=True,
                                 return_special_tokens_mask=True)
        return model_inputs
    
    # apply func on all pairs of sentences
    # executed only by the main process
    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer",
        )
        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["validation"]


    # Data Collator: pad the inputs/labels to the maximum length in the batch 
    # return tensors: pytorch
    data_collator = DataCollatorForLanguageModeling(
        tokenizer, 
        mlm_probability=0.15, 
        # pad_to_multiple_of=8 if accelerator.use_fp16 else None, 
        return_tensors="pt")

    ##### Define training/testing sets  to train model 
    # columns: which columns serving as independent variables
    # batch_size: for training 
    # shuffle: whether want to shuffle dataset
    # collate_fn: collator function 

    # integrate datasets with collator 
    train_dataloader = DataLoader(train_dataset,
                                shuffle=True, 
                                collate_fn=data_collator,
                                batch_size=args.batch_size)

    eval_dataloader = DataLoader(eval_dataset, 
                                 collate_fn=data_collator, 
                                 batch_size=args.batch_size)

    ##### BUILDING & COMPLILING THE MODEL
    # Define optimizer 
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # Learning schedule 
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # feed our model, optimizer, and dataloaders to the accelerator 
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)    

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Initialize the trackers, also store configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers(project_name='pretrain_bart_pubmed', config=experiment_config)

    ##### TRAINING & EVALUATING
    # training loop
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    es =  EarlyStopping()
    completed_steps = 0

    for epoch in range(args.num_train_epochs):
        # Training
        model.train()
        total_train_loss = 0
        # Itering over all examples in train_loader 
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                train_loss = outputs.loss
                total_train_loss += train_loss.detach().float()

                accelerator.backward(train_loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Check id the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break
        
        # Evaluation
        model.eval()
        best_eval_loss = float('inf')
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch) 
                
            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.batch_size)))
        
        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        if accelerator.is_main_process:
            logger.info(f'Epoch: {epoch}, train_loss: {train_loss}, eval_loss: {eval_loss}')

        is_best = eval_loss < best_eval_loss
        if is_best: 
            best_epoch = epoch
            best_eval_loss = eval_loss

        # Update result
        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_train_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                    "learning_rates": optimizer.param_groups[0]["lr"]
                },
                step=completed_steps,
            )
        
        if is_best:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                logger.info(f'Best model save at epoch {epoch}')
                
        if args.checkpointing_steps == "epoch" and epoch%5==0:
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir, safe_serialization=True)              
        
        # early stopping
        if es.step(eval_loss.clone().detach()): 
            # check that works with a flag set by a particular process
            accelerator.set_trigger()

        if accelerator.check_trigger():
            print(f"Stopping early after epoch {epoch}")
            break
        
    accelerator.end_training()
    # Save and upload
    # blocking=False: can push checkpoints per epoch asynchronously
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and args.push_to_hub:
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"best_eval_loss": best_eval_loss.item(), "perplexity": perplexity}, f)

            if accelerator.is_main_process and args.push_to_hub:
                api.upload_folder(
                    commit_message=f"epoch {best_epoch}",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )
                logger.info(f'***** Best model is saved at epoch {best_epoch} *****')

if __name__ == "__main__":
    wandb.login(key='1a7b141f46dd483782c656aad7957e06935592e8')
    main()