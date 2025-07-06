import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MT5ForConditionalGeneration, Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, concatenate_datasets, ClassLabel, Dataset, DatasetDict, load_from_disk
import numpy as np
import huggingface_hub
from sklearn.metrics import f1_score
import gc
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import random
import warnings
import transformers
import datasets
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings('ignore')


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def go_lower(example, columns):
    for col in columns:
        if col in example:
            try:
                example[col] = example[col].lower()
            except:
                example[col] = ''
    return example

def tokenize_LEXICON(examples, tokenizer):
    tokenized_input = tokenizer(examples['text'], padding='max_length', max_length=64, return_tensors="pt")
    
    word_ids = tokenized_input.tokens()

    return {'input_ids': tokenized_input['input_ids'][0], 'word_tokens': word_ids}

def leave_only_sentence_tokens_LEXICON(example, tokenizer):
    tokens = example['input_ids']
    word_tokens_old = example['word_tokens']
    word_tokens = []
    clean_tokens = []
    for tok, wt in list(zip(tokens, word_tokens_old)):
        if tok == tokenizer.eos_token_id:
            break
        else:
            clean_tokens.append(tok)
            word_tokens.append(wt)
    # Ensure the lists are not empty before returning
    if not clean_tokens:
        # Handle cases where the input might be empty or only contain special tokens
        # Return empty lists or handle as appropriate for your use case
        return {'clean_input_ids': [], 'clean_word_tokens': []}
    return {'clean_input_ids': clean_tokens, 'clean_word_tokens': word_tokens}


def preprocess_SYNTHDETOX(examples, tokenizer):
    inputs = []
    targets = []
    tox = examples['toxic_sentence']
    neu = examples['neutral_sentence']
    lang = examples['lang']
    lang_map = {
        'am': 'amharic',
        'ar': 'arabic',
        'de': 'german',
        'en': 'english',
        'es': 'spanish',
        'fr': 'french',
        'he': 'hebrew',
        'hi': 'hindi',
        'hin': 'hinglish',
        'it': 'italian',
        'ja': 'japanese',
        'tt': 'tatar',
        'kk': 'kazakh',
        'ru': 'russian',
        'uk': 'ukranian',
        'zh': 'chinese',
    }
    if tox:  # If toxic text is not empty
        prompt = f"Detoxify this {lang_map[lang]} text and return answer also in {lang_map[lang]}. Here is the text: {tox}"
        inputs.append(prompt)
        targets.append(neu)
    
    # Ensure inputs/targets are not empty before tokenizing
    if not inputs:
       # Handle case where toxic sentence was empty or None
       # Return empty dict or structure matching expected output
       # Adjust max_length as needed, using 128 from original code
       return {
           'input_ids': torch.zeros(128, dtype=torch.long),
           'attention_mask': torch.zeros(128, dtype=torch.long),
           'labels': torch.zeros(128, dtype=torch.long),
           'toxic_word_tokens': [''] * 128 # Placeholder, adjust as needed
       }


    model_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=128, return_tensors="pt")

    # Ensure toxic sentence is not None before tokenizing for word tokens
    toxic_sentence_for_tokens = examples['toxic_sentence'] if examples['toxic_sentence'] else ""
    word_inputs = tokenizer(toxic_sentence_for_tokens, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    word_tokens = word_inputs.tokens()
    
    # Tokenize target texts
    labels = tokenizer(targets, padding='max_length', truncation=True, max_length=128, return_tensors="pt").input_ids
    
    # Ensure correct shape if batch size is 1 from .map()
    model_inputs["input_ids"] = model_inputs["input_ids"].squeeze(0)
    model_inputs["attention_mask"] = model_inputs["attention_mask"].squeeze(0)
    model_inputs["labels"] = labels.squeeze(0)

    model_inputs['toxic_word_tokens'] = word_tokens
    
    return model_inputs

def find_overlap_position(main_array, sub_array):
    """
    Find the first position where sub_array starts in main_array.
    
    Args:
        main_array (list): The main array to search within.
        sub_array (list): The sub_array to search for.
    
    Returns:
        int: The starting index of the first occurrence of sub_array in main_array, or -1 if not found.
    """
    main_len = len(main_array)
    sub_len = len(sub_array)
    
    # Ensure sub_array is not empty and not longer than main_array
    if not sub_array or sub_len == 0 or sub_len > main_len:
        return -1
    
    for i in range(main_len - sub_len + 1):
        if main_array[i:i + sub_len] == sub_array:
            return i
    
    return -1

# Modified function signature to accept lexicon_token_list
def align_LEXICON(examples, lexicon_token_list):
    # Handle batched processing
    token_classes_batch = []
    
    for example_idx, input_tokens in enumerate(examples['toxic_word_tokens']):
        # Initialize token classes array with zeros
        token_classes = [0] * 128
        
        # Skip processing if input_tokens is empty
        if not input_tokens:
            token_classes_batch.append(token_classes)
            continue
        
        # Create a string representation of input tokens for faster substring matching
        # Ensure all tokens are strings before joining
        input_str = ' '.join([str(token) if not isinstance(token, str) else token for token in input_tokens])
        
        # Process lexicon tokens in batches to reduce overhead
        for lexicon_tokens in lexicon_token_list:
            # Skip empty lexicon token lists
            if not lexicon_tokens:
                continue
                
            # Convert lexicon tokens to string for faster matching
            lexicon_str = ' '.join(lexicon_tokens)
            
            # Skip if lexicon string is too long to be in input
            if len(lexicon_str) > len(input_str):
                continue
                
            # Use string find method which is faster than list comparison
            if lexicon_str in input_str:
                # If found, do the more expensive token-by-token matching
                res = find_overlap_position(input_tokens, lexicon_tokens)
                if res != -1:
                    # Ensure slicing does not go out of bounds
                    end_index = min(res + len(lexicon_tokens), 128)
                    # Use slice assignment which is faster than individual assignments
                    token_classes[res:end_index] = [1] * (end_index - res)
        
        token_classes_batch.append(token_classes)
    
    return {'token_classes': token_classes_batch}

def add_columns(data, tox, labse):
    def compute_sim(example):
        example['sim'] = util.cos_sim(labse.encode(example['toxic_sentence'], convert_to_tensor=True), labse.encode(example['neutral_sentence'], convert_to_tensor=True))[0][0]
        return example

    def compute_sta(batch):
        # Run the classifier on whole lists – the pipeline will chunk them
        tox_out   = tox(batch["toxic_sentence"])
        detox_out = tox(batch["neutral_sentence"])

        # Convert pipeline output to numbers we need
        sta_tox, sta_detox, sta_reduction = [], [], []
        for t, d in zip(tox_out, detox_out):
            t_score = t["score"] if t["label"] == "LABEL_0" else 1 - t["score"]
            d_score = d["score"] if d["label"] == "LABEL_0" else 1 - d["score"]

            sta_tox.append(t_score)
            sta_detox.append(d_score)
            sta_reduction.append(d_score - t_score)

        # Add the new lists to the batch‑dict
        batch["sta_tox"] = sta_tox
        batch["sta_detox"] = sta_detox
        batch["sta_reduction"] = sta_reduction
        return batch

    data = data.map(
        compute_sta,
        batched=True,
        batch_size=128,
        desc="Computing style‑transfer accuracy"
    )

    data = data.map(compute_sim)

    return data

class CreateDataset:
    def __init__(self, langs):
        self.langs = langs

    def main(self):
        mpd_full = load_dataset('textdetox/multilingual_paradetox')
        sdm_full = load_dataset('s-nlp/synthdetoxm')['train']
        synth_full = Dataset.from_pandas(pd.read_csv('./synthdetoxm_argmaxed.tsv', sep='\t'))
        synth_full = synth_full.rename_columns(
            {
                'Lang': 'lang',
                'Toxic': 'toxic_sentence',
                'Detoxed': 'neutral_sentence',
                'SIM': 'sim',
                'Toxic STA pipe': 'sta_tox',
                'STA pipe': 'sta_detox',
                'reduction_list_pipe': 'sta_reduction',
            }
        ).remove_columns(
            ['Which model', 'Toxic STA api', 'STA api']
        )
        mpd_list = []
        tokenizer = AutoTokenizer.from_pretrained('bigscience/mt0-large') # Load tokenizer once

        tox = transformers.pipeline('text-classification', 'textdetox/xlmr-large-toxicity-classifier-v2', device='cuda:0')
        labse = SentenceTransformer('sentence-transformers/LaBSE', device='cuda:1')

        for lang in self.langs:
            mpd = None
            synth = None
            sdm = None

            print(f"Processing language: {lang}")
            # Load and process mpd for this language
            
            try:
                mpd = mpd_full[lang]
                mpd = add_columns(mpd, tox, labse)
                sta_mean = mpd.to_pandas().sta_detox.mean()
                sim_mean = mpd.to_pandas().sta_reduction.mean()
                mpd = mpd.filter(lambda x: x['sim'] > sim_mean and x['sta_detox'] > sta_mean)
            except:
                pass
            
            try:
                synth = synth_full.filter(lambda x: x['lang'] == lang)
                sta_mean = synth.to_pandas().sta_detox.mean()
                sim_mean = synth.to_pandas().sta_reduction.mean()
                synth = synth.filter(lambda x: x['sim'] > sim_mean and x['sta_detox'] > sta_mean)
            except:
                pass 

            try:
                sdm = sdm_full.filter(lambda x: x['lang'] == lang)
                sdm = add_columns(sdm, tox, labse)
                sta_mean = sdm.to_pandas().sta_detox.mean()
                sim_mean = sdm.to_pandas().sta_reduction.mean()
                sdm = sdm.filter(lambda x: x['sim'] > sim_mean and x['sta_detox'] > sta_mean)
            except:
                pass

            float32_cols = ["sim", "sta_tox", "sta_detox", "sta_reduction"]

            def ensure_float32(ds, cols=float32_cols):
                """
                Return a copy of *ds* where the listed columns are float32.
                Columns that already have the right dtype are skipped.
                """
                for col in cols:
                    if col in ds.column_names and ds.features[col].dtype != "float32":
                        ds = ds.cast_column(col, datasets.Value("float32"))
                return ds
            
            try:
                mpd = ensure_float32(mpd)
            except AttributeError:
                pass

            try:
                sdm = ensure_float32(sdm)
            except AttributeError:
                pass

            try:
                synth = ensure_float32(synth)
            except AttributeError:
                pass
            
            try:
                if lang in ['de', 'es', 'ru']:
                    mpd = concatenate_datasets([mpd, sdm, synth])
                elif lang in ['en', 'ar', 'es', 'de', 'hi', 'zh', 'uk', 'am']:
                    mpd = concatenate_datasets([mpd, synth])
                elif lang == 'fr':
                    mpd = concatenate_datasets([sdm, synth])
                else:
                    mpd = synth
            except:
                print(mpd, sdm, synth)
                raise

            columns_to_lowercase = ['toxic_sentence', 'neutral_sentence']
            print(f"[{lang}] Lowercasing mpd...")
            mpd = mpd.map(lambda example: go_lower(example, columns_to_lowercase), batched=False) # Consider batched=True if go_lower handles batches
            
            # Add 'lang' column to each example
            print(f"[{lang}] Adding language column...")
            mpd = mpd.map(lambda x: {**x, "lang": lang}, batched=False) # Consider batched=True
            
            # Load and process toxic_lexicon for this language
            print(f"[{lang}] Loading toxic lexicon...")
            if lang == 'kk':
                toxic_lexicon_ds = load_dataset('textdetox/multilingual_toxic_lexicon')['ru']
            else:
                toxic_lexicon_ds = load_dataset('textdetox/multilingual_toxic_lexicon')[lang]

            # Lowercase lexicon text directly
            toxic_lexicon_ds = toxic_lexicon_ds.map(lambda x: {'text': x['text'].lower() if x['text'] else ''})

            print(f'[{lang}] Tokenizing lexicon...')
            # Use batched=True for potentially faster tokenization if tokenizer handles it well
            toxic_lexicon_ds = toxic_lexicon_ds.map(lambda x: tokenize_LEXICON(x, tokenizer), batched=False) 
            
            print(f'[{lang}] Leaving only sentence tokens lexicon...')
            # Use batched=True if function can handle batches
            toxic_lexicon_ds = toxic_lexicon_ds.map(lambda x: leave_only_sentence_tokens_LEXICON(x, tokenizer), batched=False) 
            
            # --- Optimization Start ---
            # Extract the list of clean word tokens ONCE before the main map operation
            print(f'[{lang}] Extracting lexicon token list...')
            # Filter out potential None entries if leave_only_sentence_tokens_LEXICON can return None/empty
            lexicon_token_list = [tokens for tokens in toxic_lexicon_ds['clean_word_tokens'] if tokens]
            # --- Optimization End ---

            print(f'[{lang}] Preprocessing synthdetox...')
            # Use batched=True if preprocess_SYNTHDETOX can handle batches
            mpd = mpd.map(lambda x: preprocess_SYNTHDETOX(x, tokenizer), batched=False) 
            
            print(f'[{lang}] Aligning lexicon...')
            # Pass the pre-computed list using fn_kwargs
            # Use batched processing for better performance
            mpd = mpd.map(align_LEXICON, batched=True, batch_size=32, fn_kwargs={'lexicon_token_list': lexicon_token_list})
            
            mpd_list.append(mpd)
            
            # Clean up memory
            del toxic_lexicon_ds, lexicon_token_list, mpd
            gc.collect()
        
        # Concatenate all datasets
        print("Concatenating datasets...")
        combined_dataset = concatenate_datasets(mpd_list)
        print(combined_dataset)

        # Save the combined dataset
        print("Saving dataset to disk...")
        combined_dataset.save_to_disk('with_markup_mean')
        print("Dataset creation complete.")


# python create_dataset.py --langs am ar de en es fr he hi hin it ja tt kk zh ru uk
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Create dataset with token markup.')
    
    parser.add_argument('--langs', type=str, nargs='+', required=True, help='List of langs to process')

    args = parser.parse_args()

    create_dataset = CreateDataset(args.langs)
    create_dataset.main()
