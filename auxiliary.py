import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MT5ForConditionalGeneration, Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, concatenate_datasets, ClassLabel, Dataset, DatasetDict, load_from_disk
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

with open('token.txt', 'r') as t:
    token = t.readlines()[0]
    huggingface_hub.login(token)


def write_logs(file, logs):
    with open(file, 'a') as out:
        out.write(logs + '\n')


def prepare_dataset(top_n):
    synthdetoxm_ru = load_dataset('alexandro767/synthdetoxm_ru_with_token_classes')['train']
    synthdetoxm_fr = load_dataset('alexandro767/synthdetoxm_fr_with_token_classes')['train']
    synthdetoxm_es = load_dataset('alexandro767/synthdetoxm_es_with_token_classes')['train']
    synthdetoxm_new = concatenate_datasets([synthdetoxm_ru, synthdetoxm_fr, synthdetoxm_es])
    synthdetoxm_new.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'], output_all_columns=True)
    
    def go_squeeze(examples):
        input_ids = examples['input_ids'].squeeze()
        attention_mask = examples['attention_mask'].squeeze()
        labels = examples['labels'].squeeze()
    
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        
    def align_tokens_with_pad(examples):
        token_classes = examples['token_classes']
        attention_mask = examples['attention_mask']
        
        ignore_value = -100
        ignore_tensor = torch.full_like(token_classes, ignore_value)
        
        token_classes = torch.where(attention_mask == 0, ignore_tensor, token_classes)
    
        return {'token_classes': token_classes}

    def _maybe_slice(x):
        return x if top_n == -1 else x[:top_n]
    
    synthdetoxm_new = synthdetoxm_new.map(go_squeeze, batched=True)
    synthdetoxm_new = synthdetoxm_new.map(align_tokens_with_pad, batched=True)
    
    df_for_collator = Dataset.from_dict({'toxic_sentence': _maybe_slice(synthdetoxm_new['toxic_sentence']),
                                         'neutral_sentence': _maybe_slice(synthdetoxm_new['neutral_sentence']),
                                         'lang': _maybe_slice(synthdetoxm_new['lang']),
                                         'cls_input_ids': _maybe_slice(synthdetoxm_new['input_ids']), 
                                         'cls_attention_mask': _maybe_slice(synthdetoxm_new['attention_mask']), 
                                         'cls_labels': _maybe_slice(synthdetoxm_new['token_classes']), 
                                         'detox_input_ids': _maybe_slice(synthdetoxm_new['input_ids']), 
                                         'detox_attention_mask': _maybe_slice(synthdetoxm_new['attention_mask']),
                                         'detox_labels': _maybe_slice(synthdetoxm_new['labels']), 
                                         'toxic_word_tokens': _maybe_slice(synthdetoxm_new['toxic_word_tokens'])})
    return df_for_collator


def prepare_dataset_2(path='with_markup_mean', top_n=-1):
    
    def _maybe_slice(x):
        return x if top_n == -1 else x[:top_n]

    dataset = load_from_disk(path)

    df_for_collator = Dataset.from_dict({'toxic_sentence': _maybe_slice(dataset['toxic_sentence']),
                                         'neutral_sentence': _maybe_slice(dataset['neutral_sentence']),
                                         'lang': _maybe_slice(dataset['lang']),
                                         'cls_input_ids': _maybe_slice(dataset['input_ids']), 
                                         'cls_attention_mask': _maybe_slice(dataset['attention_mask']), 
                                         'cls_labels': _maybe_slice(dataset['token_classes']), 
                                         'detox_input_ids': _maybe_slice(dataset['input_ids']), 
                                         'detox_attention_mask': _maybe_slice(dataset['attention_mask']),
                                         'detox_labels': _maybe_slice(dataset['labels']), 
                                         'toxic_word_tokens': _maybe_slice(dataset['toxic_word_tokens'])})
    return df_for_collator