import argparse

import huggingface_hub
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, TrainingArguments)

from auxiliary import prepare_dataset

import logging
import fire
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

with open('token.txt', 'r') as t:
    token = t.readlines()[0]
    huggingface_hub.login(token)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        # Perform a training step
        loss = super().training_step(model, inputs)
        
        # Optionally, you can compute and print additional metrics here
        # For example, you can use the compute_metrics method if you have defined it
        # For simplicity, we'll just print the loss
        # print(f"Training loss after step {self.state.global_step}: {loss.item()}")
        
        return loss


def baseline(model_path, output_dir, top_n, use_custom_trainer, add_sdm):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    df_for_collator = prepare_dataset(top_n)
    df_for_collator = df_for_collator.select_columns(['detox_input_ids', 'detox_attention_mask', 'detox_labels'])
    df_for_collator = df_for_collator.rename_columns({'detox_input_ids': 'input_ids', 'detox_attention_mask': 'attention_mask', 'detox_labels': 'labels'})
    train_test = df_for_collator.train_test_split(test_size=0.15, shuffle=True, seed=42)
    train_dataset = train_test['train']
    eval_dataset = train_test['test']

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy='steps',
        learning_rate=5e-5,
        optim='adafactor',
        lr_scheduler_type='cosine',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=7,
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=50,
        save_total_limit=4,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        remove_unused_columns=True,
        report_to="none",
        gradient_accumulation_steps=4,
        disable_tqdm=False,
        seed=42,
    )

    # Pass the flag to the training function
    train_detoxification_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        training_args=training_args,
        use_custom_trainer=use_custom_trainer # Pass the flag
    )

    if add_sdm:
        pass


def train_detoxification_model(model, tokenizer, train_dataset, eval_dataset, output_dir, training_args, use_custom_trainer=False):
    if use_custom_trainer:
        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    else:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

    trainer.train()
    return trainer


def prepare_and_train_current_dataset(model, tokenizer, output_dir, top_n, use_custom_trainer):
    logging.info("Preparing and training on the current dataset...")
    df_for_collator = prepare_dataset(top_n)
    df_for_collator = df_for_collator.select_columns(['detox_input_ids', 'detox_attention_mask', 'detox_labels'])
    df_for_collator = df_for_collator.rename_columns({'detox_input_ids': 'input_ids', 'detox_attention_mask': 'attention_mask', 'detox_labels': 'labels'})
    train_test = df_for_collator.train_test_split(test_size=0.15, shuffle=True, seed=42)
    train_dataset = train_test['train']
    eval_dataset = train_test['test']

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy='steps',
        learning_rate=5e-5,
        optim='adafactor',
        lr_scheduler_type='cosine',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=7,
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=50,
        save_total_limit=4,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        remove_unused_columns=True,
        report_to="none",
        gradient_accumulation_steps=4,
        disable_tqdm=False,
        seed=42,
    )

    train_detoxification_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        training_args=training_args,
        use_custom_trainer=use_custom_trainer
    )


def main(model_path="bigscience/mt0-xl", output_dir="./output", top_n=-1, use_custom_trainer=False, add_sdm=False):
    logging.info("Starting two-stage training process...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    logging.info("Starting Stage 1: Training on s-nlp/synthdetoxm dataset...")
    synthdetoxm_dataset = load_dataset("s-nlp/synthdetoxm")

    def preprocess_synthdetoxm(examples):
        inputs = examples['toxic_sentence']
        targets = examples['neutral_sentence']
        model_inputs = tokenizer(inputs, max_length=256, truncation=True)
        labels = tokenizer(targets, max_length=256, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    processed_synthdetoxm_dataset = synthdetoxm_dataset.map(preprocess_synthdetoxm, batched=True)
    processed_synthdetoxm_dataset = processed_synthdetoxm_dataset.select_columns(['input_ids', 'attention_mask', 'labels'])

    synthdetoxm_train_dataset = processed_synthdetoxm_dataset['train']
    train_test_synthdetoxm = synthdetoxm_train_dataset.train_test_split(test_size=0.1, seed=42)
    synthdetoxm_train_dataset = train_test_synthdetoxm['train']
    synthdetoxm_eval_dataset = train_test_synthdetoxm['test']

    stage1_output_dir = os.path.join(output_dir, "stage1_synthdetoxm")
    stage1_training_args = Seq2SeqTrainingArguments(
        output_dir=stage1_output_dir,
        eval_strategy="epoch",
        save_strategy='epoch', # Save checkpoint after each epoch for easy loading
        learning_rate=5e-5,
        optim='adafactor',
        lr_scheduler_type='cosine',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=50,
        save_total_limit=1,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        remove_unused_columns=True,
        report_to="none",
        gradient_accumulation_steps=4,
        disable_tqdm=False,
        seed=42,
    )

    trainer = train_detoxification_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=synthdetoxm_train_dataset,
        eval_dataset=synthdetoxm_eval_dataset,
        output_dir=stage1_output_dir,
        training_args=stage1_training_args,
        use_custom_trainer=use_custom_trainer # Pass the flag
    )

    logging.info("Stage 1 training complete. Loading model from checkpoint...")
    best_stage1_checkpoint = trainer.state.best_model_checkpoint # This might not be available directly, need to find the best checkpoint path
    model = AutoModelForSeq2SeqLM.from_pretrained(stage1_output_dir)

    logging.info("Starting Stage 2: Training on the current dataset...")
    stage2_output_dir = os.path.join(output_dir, "stage2_current_dataset")

    stage2_training_args = Seq2SeqTrainingArguments(
        output_dir=stage2_output_dir,
        eval_strategy="epoch",
        save_strategy='epoch',
        learning_rate=5e-5,
        optim='adafactor',
        lr_scheduler_type='cosine',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=7,
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=50,
        save_total_limit=4,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        remove_unused_columns=True,
        report_to="none",
        gradient_accumulation_steps=4,
        disable_tqdm=False,
        seed=42,
    )

    prepare_and_train_current_dataset(
        model=model,
        tokenizer=tokenizer,
        output_dir=stage2_output_dir,
        top_n=top_n,
        use_custom_trainer=use_custom_trainer
    )

    logging.info("Two-stage training process finished.")
    
    if add_sdm:
        pass # Keep this for now, although its purpose is unclear and wasn't part of the two-stage plan.

if __name__ == '__main__':
    fire.Fire(main)
