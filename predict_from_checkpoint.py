import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MT5ForConditionalGeneration, Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, concatenate_datasets, ClassLabel, Dataset, DatasetDict
import numpy as np
import huggingface_hub
from sklearn.metrics import f1_score
import gc
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import argparse
import warnings
warnings.filterwarnings('ignore')

from auxiliary import prepare_dataset, write_logs
from two_losses import TwoLossesModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def token2classes(data_path):
    if data_path[-1] != '/':
        data_path += '/'
        
    out_file = data_path+'token2classes.txt'
    
    with open(data_path+'toxic_word_tokens.txt', 'r') as f:
        token_lines = f.readlines()
    with open(data_path+'classes.txt', 'r') as f:
        classes_lines = f.readlines()

    for tokens, classes in list(zip(token_lines, classes_lines)):
        with open(out_file, 'a') as f:
            tokens_sp = tokens.split(', ')
            classes_sp = classes.split(', ')
            for t, c in list(zip(tokens_sp, classes_sp)):
                line = f'{t}, {c}'
                f.write(line+'\n')
            f.write('\n')


def prepare_item(lang, tox):
    lang_map = {
        'en': 'english',
        'ru': 'russian',
        'uk': 'ukranian',
        'de': 'german',
        'es': 'spanish',
        'am': 'amharic',
        'zh': 'chinese',
        'ar': 'arabic',
        'hi': 'hindi',
        'it': 'italian',
        'fr': 'french',
        'he': 'hebrew',
        'hin': 'hinglish',
        'ja': 'japanese',
        'tt': 'tatar',
    }

    return f"Detoxify this {lang} text and return answer also in {lang_map[lang]}. Here is the text: {tox}"


class PredictFromCheckpoint:
    def __init__(self, checkpoint_path, model_name, top_n, dataset_save_path):
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.top_n = top_n
        self.dataset_save_path = dataset_save_path

    def main(self):
        # df_for_collator = prepare_dataset(self.top_n)
        
        # train_test = df_for_collator.train_test_split(test_size=0.15, shuffle=True, seed=42)
        # eval_dataset = train_test['test']  # Only need eval dataset for prediction
        eval_dataset = load_dataset('textdetox/multilingual_paradetox_test')
        for lang in eval_dataset:
            eval_dataset[lang] = eval_dataset[lang].add_column('lang', [lang,] * len(eval_dataset[lang]))

        eval_dataset = concatenate_datasets([eval_dataset[_] for _ in eval_dataset])
        eval_dataset = eval_dataset.rename_columns({'text': 'toxic_sentence'})

        model_cpt = (
            TwoLossesModel.from_pretrained(self.checkpoint_path).to(device) 
            if 'tl' in self.checkpoint_path 
            else AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint_path)
            ).to(device)
        model_cpt.eval()
        tokenizer_cpt = AutoTokenizer.from_pretrained(self.model_name)
        
        eval_dataset.set_format(
            type='torch', 
            columns=['toxic_sentence', 'lang'], 
            output_all_columns=True,
        )
        
        eval_bs = 16
        detoxified_sentence = []

        # Generating detoxified sentences
        for i in tqdm(range(0, len(eval_dataset), eval_bs)):
            batch_combined = eval_dataset[i:i + eval_bs]

            batch_toxic_sentence = batch_combined['toxic_sentence']
            batch_lang = batch_combined['lang']
            
            input_ids = tokenizer_cpt([prepare_item(l, bt) for bt, l in zip(batch_toxic_sentence, batch_lang)], padding='max_length', truncation=True, max_length=128, return_tensors='pt')

            with torch.no_grad():
                outputs = model_cpt.generate(
                    input_ids=input_ids['input_ids'].to(device), 
                    attention_mask=input_ids['attention_mask'].to(device), 
                    max_length=128
                )
            detoxified_outputs = tokenizer_cpt.batch_decode(outputs, skip_special_tokens=True)
            detoxified_sentence.extend(detoxified_outputs)  # Use extend instead of append for lists
        
        eval_dataset = eval_dataset.add_column('neutral_sentence', detoxified_sentence)
        eval_dataset = eval_dataset.select_columns(['toxic_sentence', 'neutral_sentence', 'lang'])

        eval_dataset.to_csv(self.dataset_save_path + 'toxic.tsv', sep='\t', index=None)
        
        print('Datasets saved')

if __name__ == '__main__':

    # python predict_from_checkpoint.py --checkpoint_path='base_res_v2/checkpoint-280' --model_name='bigscience/mt0-large' --top_n=-1

    parser = argparse.ArgumentParser(description='Predict detoxified sentences from a trained checkpoint.')
    
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained checkpoint.')
    parser.add_argument('--model_name', type=str, required=True, help='Name or path of the pre-trained model.')
    parser.add_argument('--top_n', type=int, default=-1, help='Number of samples to process.')

    args = parser.parse_args()

    predictor = PredictFromCheckpoint(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        top_n=args.top_n,
        dataset_save_path='/'.join(args.checkpoint_path.split('/')[:-1])+'/'
    )
    predictor.main()
