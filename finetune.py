import os
import numpy as np
import math
import json
from tqdm.auto import tqdm
import logging
import argparse
from pathlib import Path
import random
import time
import wandb
import statistics

import datasets
from datasets import load_dataset
from huggingface_hub import HfApi
import transformers
from transformers.utils import send_example_telemetry
from transformers import BartForConditionalGeneration

import torch

from accelerate import Accelerator
from accelerate.utils import set_seed

from filelock import FileLock

import sys
import nltk
import argparse
from transformers import SchedulerType

from datasets import DatasetDict
from transformers import BartTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader

from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate
from torch.optim import AdamW

from early_stopping import * 
import torch.distributed as dist
from datetime import timedelta

if torch.cuda.device_count() > 1:
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))

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
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. "
            "during ``evaluate`` and ``predict``."
        ),
    )   
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",    
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
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
                        default=None, 
                        help="Where to store the final model.")
    parser.add_argument("--seed", 
                        type=int, 
                        default=11, 
                        help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", default=True, help="Whether or not to push the model to the Hub.")
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
        default='epoch',
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--with_tracking",
        default=True,
        action="store_true",
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
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument(
        "--finetune_case",
        type=int,
        default="0",
        help="""5 cases: 
            0. model: bart-base, input: OCRtext 
            1. model: bart-cdip, input: OCRtext
            2. model: bart-cdip, input: QAtext
            3. model: bart-cdip, input: question_text
            4. model: bart-cdip, input: answer_text"""
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument("--model_checkpoint", type=str, help="Model identifier from huggingface.co/models.", required=False,)

    args = parser.parse_args()
    
    if args.finetune_case == 0: 
        args.model_dir = 'cppmai/pretrain_bart_pubmed'
        args.output_dir = 'bart-sum_pubmed'
    # elif args.finetune_case == 1: 
    #     args.model_dir = 'cppmai/pretrained_bart_cdip'
    #     args.output_dir = 'Msum_bart-cdip_OCRtext'
    # elif args.finetune_case == 2: 
    #     args.model_dir = 'cppmai/pretrained_bart_cdip'
    #     args.output_dir = 'Msum_bart-cdip_QAtext'
    # elif args.finetune_case == 3: 
    #     args.model_dir = 'cppmai/pretrained_bart_cdip'
    #     args.output_dir = 'Msum_bart-cdip_question-text'
    # elif args.finetune_case == 4: 
    #     args.model_dir = 'cppmai/pretrained_bart_cdip'
    #     args.output_dir = 'Msum_bart-cdip_anwser_text'
    return args

# ROUGE metric expect: generated summaries into sentences that are separated by newlines
def postprocess_text(preds, labels):
    try:
        nltk.data.find('tokenizers/punkt')
    except (LookupError, OSError):
        with FileLock('.lock') as lock:
            nltk.download('punkt', quiet=True)    
            
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def setup_logger(log_dir):
    # Set up logger
    logger = logging.getLogger('main_logger')
    logger.setLevel(logging.DEBUG)  # Set log level

    # Check if logger already has handlers
    if not logger.hasHandlers():  # Prevents adding handlers multiple times
        # Create file handler that logs to a specific file
        log_filename = os.path.join(log_dir, 'main.log')
        fh = logging.FileHandler(log_filename)
        fh.setLevel(logging.DEBUG)

        # Create formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(fh)
    return logger

# Redirect output to files
def redirect_output(log_dir):
    sys.stdout = open(os.path.join(log_dir, 'main_stdout.log'), 'w')
    sys.stderr = open(os.path.join(log_dir, 'main_stderr.log'), 'w')

def main():     
    ##### Initialize accelerator
    logger.info('Initialize accelerator')
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs) 

    # make 1 log on every process with configuration for debugging   
    logger.info(accelerator.state)

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

    ##### HuggingFace repo
    if accelerator.is_main_process and args.push_to_hub:
        # Retrieve of infer repo_name
        repo_name = 'finetune_bart-sum_pubmed'
        # Create repo and retrieve repo_id
        api = HfApi()
        repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id
        logger.info('HuggingFace folder is created')
        
        with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
            if "step_*" not in gitignore:
                gitignore.write("step_*\n")
            if "epoch_*" not in gitignore:
                gitignore.write("epoch_*\n")
    accelerator.wait_for_everyone()

    ##### Download 
    # dataset
    logger.info('Load data')
    raw_datasets = load_dataset('cppmai/finetune_pubmed')
    # raw_datasets["train"] = raw_datasets["train"].select(range(50))
    # raw_datasets["validation"] = raw_datasets["validation"].select(range(10))
    # raw_datasets["test"] = raw_datasets["test"].select(range(10))

    # model 
    model = BartForConditionalGeneration.from_pretrained(
        args.model_dir, 
        trust_remote_code=args.trust_remote_code)
    # tokenizer
    tokenizer = BartTokenizer.from_pretrained(
        'facebook/bart-base',  
        trust_remote_code=args.trust_remote_code)
    
    # metric
    rouge_score = evaluate.load("rouge")
    
    ##### Pre-processing data
    def preprocess_function(examples):
        # Tokenize the texts
        if args.finetune_case == 2:
            inputs = [f'Question: {q} Answer: {a}\n Document: {c}' for q, a, c in zip(examples['question'], examples['answer'], examples['content'])]
        # elif args.finetune_case == 3:
        #     inputs = [f'Question: {q}\nDocument: {c}' for q, c in zip(examples['question'], examples['content'])]
        # elif args.finetune_case == 4:
        #     inputs = [f'Answer: {a}\n Document: {c}' for a, c in zip(examples['answer'], examples['content'])]
        else: # [0, 1]
            inputs = [f'Document: {c}' for  c in examples['article']]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, truncation=True)

        # tokenizer for targets
        labels = tokenizer(text_target = examples['abstract'], max_length=args.max_target_length, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        
        return model_inputs
    
    # tokenize input sequence and label 
    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns= raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    # dynamic padding 
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        model=model, 
        pad_to_multiple_of=8 if accelerator.use_fp16 else None, 
        return_tensors="pt")
    
    # Define training/testing sets  to train model 
    # columns: which columns serving as independent variables
    # batch_size: for training 
    # shuffle: whether want to shuffle dataset
    # collate_fn: collator function 
    train_dataloader = DataLoader(processed_datasets['train'], shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(processed_datasets['validation'], collate_fn=data_collator, batch_size=args.batch_size)  
    test_dataloader = DataLoader(processed_datasets['test'], shuffle=False, collate_fn=data_collator, batch_size=args.batch_size)
    
    ##### Hyperparameters
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

    ##### Initialize the trackers
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]
        accelerator.init_trackers(project_name="finetune_bart-sum_pubmed", 
                                  config=experiment_config)   
        
    ##### TRAINING & EVALUATING
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    rank = accelerator.process_index

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, desc=f"Process {rank}")
    es = EarlyStopping()
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    best_eval_loss = float('inf')
    for epoch in range(starting_epoch, args.num_train_epochs):
        # Training
        model.train()
        total_train_loss = 0
        total_eval_loss = 0
        
        # Itering over all examples in train_loader 
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
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
                    accelerator.save_state(output_dir, safe_serialization=True)

            if completed_steps >= args.max_train_steps:
                break
        
        # Evaluation
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                val_loss = outputs.loss
                total_eval_loss += val_loss.detach().float()

                # generate summaries for each epoch 
                # gen token
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length = args.max_target_length
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]

                # If we did not pad to max length, we need to pad the labels too
                labels = accelerator.pad_across_processes(
                    batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
                )

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                # Replace -100 in the labels as we can't decode them
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                # decode them into text 
                decoded_preds = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(
                    decoded_preds, decoded_labels
                )

                rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

        # Extract the median ROUGE scores
        result = rouge_score.compute(use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        
        # Update result
        if args.with_tracking:
            result["train_loss"] = total_train_loss.item() / len(train_dataloader)
            result["eval_loss"] = total_eval_loss.item() / len(eval_dataloader)
            result["epoch"] = epoch
            result["step"] = completed_steps 
            accelerator.log(result, step=completed_steps) 

        if accelerator.is_local_main_process:
            logger.info(f'Epoch {epoch}: {result}')
        
        is_best = result["eval_loss"] < best_eval_loss
        if is_best:
            best_epoch = epoch
            best_eval_loss = result["eval_loss"]
            best_result = result
            
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
            
            logger.info(f'Model is saved at epoch {epoch}')

        # early stopping
        if es.step(torch.tensor(result["eval_loss"])): 
            # check that works with a flag set by a particular process
            accelerator.set_trigger()
        if accelerator.check_trigger():
            print(f"Stopping early after epoch {epoch}")
            break
            
        if args.checkpointing_steps == "epoch" and epoch%5==0:
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir, safe_serialization=True)  
    
    ##### Test: run predictions on the test set at the very end
    test_predictions = []
    test_labels = []
    for step, batch in enumerate(test_dataloader):
        batch.to(accelerator.device)
        with torch.no_grad():
            outputs = model(**batch)
        # generate summaries for each epoch 
        # gen token
        generated_tokens = accelerator.unwrap_model(model).generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length = args.max_target_length)
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
        
        labels = batch["labels"]
        # If we did not pad to max length, we need to pad the labels too
        labels = accelerator.pad_across_processes(
            batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

        generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
        labels = accelerator.gather(labels).cpu().numpy()

        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        # decode them into text 
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        
        test_predictions.append(decoded_preds)
        test_labels.append(decoded_labels)

    # Save and upload
    # blocking=False: can push checkpoints per epoch asynchronously
    if args.output_dir is not None:
        accelerator.wait_for_everyone()    
        if accelerator.is_main_process and args.push_to_hub:
            test_predictions = np.concatenate(test_predictions)
            test_labels = np.concatenate(test_labels)
            np.save(os.path.join(args.output_dir,'test_predictions.npy'), test_predictions, allow_pickle=True)
            logger.info(f'Saved predictions, {len(test_predictions)}')
            np.save(os.path.join(args.output_dir,'test_labels.npy'), test_labels, allow_pickle=True)
            logger.info(f'Saved labels, {len(test_labels)}')

            all_results = {"best_result": best_result,
                           "last_result": result,
                           "best_epoch": best_epoch, 
                           "best_loss": best_eval_loss,
                           "test_result": rouge_score.compute(predictions=test_predictions, references=test_labels)}
            logger.info(all_results)
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                    json.dump(all_results, f)

            api.upload_folder(
                commit_message=f"Best epoch {best_epoch}",
                folder_path=args.output_dir,
                repo_id=repo_id,
                repo_type="model",
                token=args.hub_token)
            logger.info(f'***** Best model is saved at epoch {best_epoch} *****')
    
    accelerator.end_training()

if __name__ == "__main__":
    wandb.login(key='1a7b141f46dd483782c656aad7957e06935592e8')
            
    args = parse_args()
    send_example_telemetry("finetune_bart_classification", args)
    
    # Define paths
    base_dir = 'outputs'
    args.output_dir = os.path.join(base_dir, args.output_dir)
    log_dir = os.path.join(base_dir, f'logs/{Path(args.output_dir).absolute().name}_logs')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    logger = setup_logger(log_dir)  # Initialize the logger
#     redirect_output(log_dir)  # Redirect stdout and stderr to log files
    
    try:
        main()
        logger.info("File processed successfully.")
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        raise