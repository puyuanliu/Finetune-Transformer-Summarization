# -----------------------------------------------------------
# Load train/validation/test data from local disk
# (C) 2021 Puyuan Liu, Department of Computing Science, University of Alberta
# Released under GNU Public License (GPL)
# email puyuan@ualberta.ca
# -----------------------------------------------------------

from typing import Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from utils.helper import load_model_from_directory
from modules.summarization_dataset import GPT2Dataset


def get_model_and_tokenizer(args: argparse.Namespace, model_dir: str) -> Tuple[any, any]:
    """
    This function loads the appropriate (base) Transformer model and its corresponding tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)  # Load the tokenizer corresponding to the used model
    if "gpt2" in args.model_name:  # If GPT2 is in the model name, we assume the desired model is a gpt2 variant model
        print("Loading model...\n")
        if args.resume_training:
            transformer_model = load_model_from_directory(AutoModelForCausalLM, model_dir)
        else:
            transformer_model = AutoModelForCausalLM.from_pretrained(
            args.model_name)  # Load the transformer model from HuggingFace
        # Add some special tokens to the gpt2 model to help summarization task
        extra_special_tokens = {'pad_token': '<|pad|>', 'sep_token': '<|sep|>','unk_token': '<|unk|>'}
        tokenizer.add_special_tokens(extra_special_tokens)  # add special tokens to the gpt2 tokenizer
    else:
        # we haven't implemented other models for summarization task
        raise NotImplementedError
    transformer_model.resize_token_embeddings(len(tokenizer))  # Resize the embedding layer of the model to fit the new vocab (we added few special tokens)
    transformer_model.to(args.device)  # Move the model to desired device
    return transformer_model, tokenizer


def load_dataset(args:argparse.Namespace, tokenizer: any):
    """
    This function tokenize the desired dataset (specified in args) and load it into specific format (depending on the
    used Transformer model)
    """
    if "gpt2" in args.model_name:
        training_dataset, validation_dataset, giga_test_dataset = load_gpt2_dataset(args, tokenizer)  # Load the gpt2 dataset
    else:
        raise NotImplementedError
    return training_dataset, validation_dataset, giga_test_dataset


def load_gpt2_dataset(args: argparse.Namespace, tokenizer: any) -> Tuple[GPT2Dataset, GPT2Dataset, GPT2Dataset]:
    """
    Load the training, validation and test dataset from local disk to RAM
    """
    print("Start building dataset...\n")
    training_dataset = GPT2Dataset(article=args.train_file_dir + "article.txt",
                                   summaries=args.train_file_dir + "HC_summaries.txt", tokenizer=tokenizer,
                                   summary_length=args.summary_length, max_length=args.max_length)

    validation_dataset = GPT2Dataset(article=args.val_file_dir + "article.txt",
                                     summaries=args.val_file_dir + "HC_summaries.txt", tokenizer=tokenizer,
                                     summary_length=args.summary_length, max_length=args.max_length)

    test_dataset = GPT2Dataset(article=args.test_file_dir + "article.txt",
                               summaries=args.test_file_dir + "ref_summaries.txt", tokenizer=tokenizer,
                               summary_length=args.summary_length, max_length=args.max_length)

    return training_dataset, validation_dataset, test_dataset

