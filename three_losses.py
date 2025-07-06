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

from auxiliary import *

with open('token.txt', 'r') as t:
    token = t.readlines()[0]
    huggingface_hub.login(token)

import os
# os.environ["NCCL_DEBUG"] = "INFO"
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"
# os.environ["NCCL_SHM_DISABLE"] = "1"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer


class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct=util.cos_sim) -> None:
        super().__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Compute the embeddings and distribute them to anchor and candidates (positive and optionally negatives)
        # embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        embeddings = []

        m = self.model.module if hasattr(self.model, "module") else self.model

        for sf in sentence_features:
            out = m.encoder(input_ids=sf['input_ids'], attention_mask=sf['attention_mask'], return_dict=True)
            emb = out.last_hidden_state.mean(dim=1)
            embeddings.append(emb)
        anchors = embeddings[0]  # (batch_size, embedding_dim)
        candidates = torch.cat(embeddings[1:])  # (batch_size * (1 + num_negatives), embedding_dim)

        # For every anchor, we compute the similarity to all other candidates (positives and negatives),
        # also from other anchors. This gives us a lot of in-batch negatives.
        scores = self.similarity_fct(anchors, candidates) * self.scale
        # (batch_size, batch_size * (1 + num_negatives))

        # anchor[i] should be most similar to candidates[i], as that is the paired positive,
        # so the label for anchor[i] is i
        range_labels = torch.arange(0, scores.size(0), device=scores.device)

        return self.cross_entropy_loss(scores, range_labels)

    def get_config_dict(self) -> dict[str, Any]:
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.name}


class ThreeLossesModel(MT5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.classification_head = nn.Linear(config.d_model, 2)
        

class CustomTrainer(Trainer):
    def __init__(self, weight=1.0, losses='all', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classification_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.weight = weight 
        self.losses = losses
        self.contrastive_loss_fn = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        # Создаем loss-функцию, если она еще не была создана
        if self.contrastive_loss_fn is None:
            self.contrastive_loss_fn = MultipleNegativesRankingLoss(model)

        m = model.module if hasattr(model, "module") else model
        
        if model.training:
            cls_input_ids = inputs.get("cls_input_ids")
            cls_attention_mask = inputs.get("cls_attention_mask")
            cls_labels = inputs.get("cls_labels")
            detox_input_ids = inputs.get("detox_input_ids")
            detox_attention_mask = inputs.get("detox_attention_mask")
            detox_labels = inputs.get("detox_labels")

            # detox loss
            detox_loss = torch.tensor(0.0, device=detox_input_ids.device)
            detox_outputs = None
            
            detox_outputs = model(input_ids=detox_input_ids, attention_mask=detox_attention_mask, labels=detox_labels)
            detox_loss = detox_outputs.loss
            detox_loss = torch.mean(detox_loss, axis=0)
    
            # classification loss
            classification_loss = torch.tensor(0.0, device=detox_input_ids.device)

            encoder_outputs = m.encoder(input_ids=cls_input_ids, attention_mask=cls_attention_mask, return_dict=True)
            classification_logits = m.classification_head(encoder_outputs.last_hidden_state)
    
            classification_preds = classification_logits.view(-1, classification_logits.shape[-1])
            classification_labels = cls_labels.view(-1)
            
            classification_loss = self.classification_loss_fn(classification_preds, classification_labels)

            # contrastive loss
            contrastive_loss = torch.tensor(0.0, device=detox_input_ids.device)

            sentence_features = [
                {'input_ids': detox_input_ids, 'attention_mask': detox_attention_mask},
                {'input_ids': detox_labels, 'attention_mask': detox_attention_mask}
            ]

            contrastive_loss = self.contrastive_loss_fn(sentence_features, labels=None)

            # losses
            total_loss = torch.tensor(0.0, device=detox_input_ids.device)

            if self.losses == "detox":
                total_loss = detox_loss
                logs = f'{detox_loss}'
            elif self.losses == "detox_classification":
                total_loss = detox_loss + classification_loss
                logs = f'{detox_loss}, {classification_loss}'
            elif self.losses == "detox_contrastive":
                total_loss = detox_loss + contrastive_loss
                logs = f'{detox_loss}, {contrastive_loss}'
            else:  # 'all'
                if self.weight != 1.:
                    total_loss = (1. - self.weight) * detox_loss + self.weight / 2. * classification_loss + self.weight / 2. * contrastive_loss
                else:
                    total_loss = detox_loss + classification_loss + contrastive_loss
                logs = f'{detox_loss}, {classification_loss}, {contrastive_loss}'
                
            print(logs)
            
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
            
            inputs = {k: torch.tensor(v) if isinstance(v, list) else v for k, v in inputs.items()}
            
            detox_input_ids = inputs.get("input_ids")
            detox_attention_mask = inputs.get("attention_mask")
            detox_labels = inputs.get("labels")
            
            detox_loss = torch.tensor(0.0, device=detox_input_ids.device)
            detox_outputs = None
            
            if detox_labels is not None:
                detox_outputs = self.model(input_ids=detox_input_ids, attention_mask=detox_attention_mask, labels=detox_labels)
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
        eval_output =  super().evaluate(ds, ignore_keys=kwargs['ignore_keys'])

        # Добавляем метрику вручную, если её нет
        if 'eval_loss' not in eval_output:
            eval_loss = self.compute_loss(self.model, ds[0])  # костыль, если нужно руками посчитать
            eval_output['eval_loss'] = eval_loss.item() if isinstance(eval_loss, torch.Tensor) else eval_loss
    
        return eval_output


class ThreeLosses:

    def __init__(self, model_path, output_dir, weight, top_n, losses):
        self.model_path = model_path
        self.output_dir = output_dir
        if self.output_dir[-1] != '/':
            self.output_dir += '/'
        self.weight = weight
        self.top_n = top_n
        self.losses = losses
    
    def main(self):
    
        df_for_collator = prepare_dataset_2(top_n=self.top_n)
        train_test = df_for_collator.train_test_split(test_size=0.15, shuffle=True, seed=42)
        train_dataset = train_test['train']
        train_dataset = train_dataset.remove_columns(['toxic_word_tokens'])
        eval_dataset = train_test['test']
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = ThreeLossesModel.from_pretrained(self.model_path)
        model.classification_head.weight.data.normal_(mean=0.0, std=0.02)
        if model.classification_head.bias is not None:
            model.classification_head.bias.data.zero_()

        # Оборачиваем в DataParallel
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        #     model = nn.DataParallel(model)
    
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            save_strategy='epoch',
            learning_rate=1e-3,
            optim='adafactor',
            lr_scheduler_type='cosine',
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=7,
            weight_decay=0.01,
            warmup_steps=50,
            logging_steps=50,
            save_total_limit=4,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            remove_unused_columns=False,
            report_to="none",
            gradient_accumulation_steps=64,
            disable_tqdm=True,
            seed=42,
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,   
            weight=self.weight,
            losses=self.losses,
        )
        
        trainer.train()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a ThreeLossesModel with detoxification and classification tasks.')
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for saving checkpoints.')
    parser.add_argument('--weight', type=float, default=1., help='Weight for the classification loss.')
    parser.add_argument('--top_n', type=int, default=-1, help='First N rows of dataset.')
    parser.add_argument('--losses', type=str, default='all', choices=['all', 'detox', 'detox_classification', 'detox_contrastive'],
                        help='Which losses to include during training.')

    args = parser.parse_args()

    model_path = args.model_path
    output_dir = args.output_dir
    weight = args.weight if args.weight != -1 else None
    top_n = args.top_n
    losses = args.losses

    three_losses = ThreeLosses(model_path, output_dir, weight, top_n, losses)
    three_losses.main()