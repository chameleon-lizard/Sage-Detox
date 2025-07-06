import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MT5ForConditionalGeneration, Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, concatenate_datasets, ClassLabel, Dataset, DatasetDict
import numpy as np
import huggingface_hub
from sklearn.metrics import f1_score
import sys
import gc
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import warnings
import argparse
warnings.filterwarnings('ignore')

from auxiliary import prepare_dataset, write_logs

with open('token.txt', 'r') as t:
    token = t.readlines()[0]
    huggingface_hub.login(token)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


class TwoLossesModel(MT5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.classification_head = nn.Linear(config.d_model, 2)


class CustomTrainer(Trainer):
    def __init__(self, *args, weight=1.0,  **kwargs):
        super().__init__(*args, **kwargs)
        self.classification_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.weight = weight 

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        m = model.module if hasattr(model, "module") else model
        
        if model.training:
            cls_input_ids = inputs.get("cls_input_ids")
            cls_attention_mask = inputs.get("cls_attention_mask")
            cls_labels = inputs.get("cls_labels")
            detox_input_ids = inputs.get("detox_input_ids")
            detox_attention_mask = inputs.get("detox_attention_mask")
            detox_labels = inputs.get("detox_labels")
        
            detox_loss = torch.tensor(0.0, device=detox_input_ids.device)
            detox_outputs = None
            if detox_labels is not None:
                detox_outputs = model(input_ids=detox_input_ids, attention_mask=detox_attention_mask, labels=detox_labels)
                detox_loss = detox_outputs.loss
                detox_loss = torch.mean(detox_loss, axis=0)
    
            
            classification_loss = torch.tensor(0.0, device=detox_input_ids.device)
            if cls_labels is not None:

                encoder_outputs = m.encoder(input_ids=cls_input_ids, attention_mask=cls_attention_mask, return_dict=True)
                classification_logits = m.classification_head(encoder_outputs.last_hidden_state)
        
                classification_preds = classification_logits.view(-1, classification_logits.shape[-1])
                classification_labels = cls_labels.view(-1)
                
                classification_loss = self.classification_loss_fn(classification_preds, classification_labels)
        
            if self.weight != 1.:
                total_loss = (1.-self.weight) * detox_loss + self.weight * classification_loss
            else:
                total_loss = detox_loss + classification_loss

            logs = f'{detox_loss}, {classification_loss}'
            # print(logs)
            
            if return_outputs:
                return total_loss, detox_outputs

            del cls_input_ids
            del cls_attention_mask
            del cls_labels
            del detox_input_ids
            del detox_attention_mask
            del detox_labels

            del detox_outputs
            del encoder_outputs
            del classification_logits
            del classification_preds
            del classification_labels

            return total_loss
                
        else:
            detox_input_ids = inputs.get("input_ids")
            detox_attention_mask = inputs.get("attention_mask")
            detox_labels = inputs.get("labels")
            
            detox_loss = torch.tensor(0.0, device=detox_input_ids.device)
            detox_outputs = None
            
            if detox_labels is not None:
                detox_outputs = model(input_ids=detox_input_ids, attention_mask=detox_attention_mask, labels=detox_labels)
                detox_loss = detox_outputs.loss
                detox_loss = torch.mean(detox_loss, axis=0)
    
            if return_outputs:
                return detox_loss, detox_outputs

            del detox_input_ids
            del detox_attention_mask
            del detox_labels

            del detox_outputs
            
            return detox_loss
    
    def evaluate(self, *args, **kwargs):
        ds = self.eval_dataset
        ds = ds.select_columns(['detox_input_ids', 'detox_attention_mask', 'detox_labels'])
        ds = ds.rename_columns({'detox_input_ids': 'input_ids', 'detox_attention_mask': 'attention_mask', 'detox_labels': 'labels'})
        return super().evaluate(ds, ignore_keys=kwargs['ignore_keys'])


class TwoLosses:
    def __init__(self, model_path, output_dir, weight, lr, top_n):
        self.model_path = model_path
        self.output_dir = output_dir
        if self.output_dir[-1] != '/':
            self.output_dir += '/'
        self.weight = weight
        self.lr = lr
        self.top_n = top_n
    
    def main(self):
    
        df_for_collator = prepare_dataset(self.top_n)
        train_test = df_for_collator.train_test_split(test_size=0.15, shuffle=True, seed=42)
        train_dataset = train_test['train']
        train_dataset = train_dataset.remove_columns(['toxic_word_tokens'])
        eval_dataset = train_test['test']
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = TwoLossesModel.from_pretrained(self.model_path)
        model.classification_head.weight.data.normal_(mean=0.0, std=0.02)
        if model.classification_head.bias is not None:
            model.classification_head.bias.data.zero_()

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            save_strategy='steps',
            learning_rate=self.lr,
            optim='adafactor',
            lr_scheduler_type='cosine',
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=7,
            weight_decay=0.01,
            warmup_steps=50,
            logging_steps=1,
            save_total_limit=4,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            remove_unused_columns=False,
            report_to="none",
            gradient_accumulation_steps=16,
            disable_tqdm=True,
            seed=42,
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            weight=self.weight,  # Pass the weight parameter
        )
        
        trainer.train()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a TwoLossesModel with detoxification and classification tasks.')
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for saving checkpoints.')
    parser.add_argument('--weight', type=float, default=1., help='Weight for the classification loss.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--top_n', type=int, default=-1, help='First N rows of dataset.')

    args = parser.parse_args()

    model_path = args.model_path
    output_dir = args.output_dir
    weight = args.weight if args.weight != -1 else None
    lr = args.lr
    top_n = args.top_n

    two_losses = TwoLosses(model_path, output_dir, weight, lr, top_n)
    two_losses.main()
