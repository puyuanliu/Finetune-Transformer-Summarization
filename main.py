#!/usr/bin/env python
# -----------------------------------------------------------
# Wrapper function to fine-tune transformer models from Huggingface with the HC summary
# from the Gigaword dataset.
# This file serves as the main file for the project "unsupervised sentence summarization"
# which is supervised by Dr. Mou (Lili Mou).
# (C) 2021 Puyuan Liu, Department of Computing Science, University of Alberta
# Released under GNU Public License (GPL)
# Email puyuan@ualberta.ca
# -----------------------------------------------------------

from utils.helper import initialize_wandb_environment, load_model_from_directory, generate_model_dir
from utils.options import parse_argument, get_training_argument
from utils.load_data import get_model_and_tokenizer, load_dataset
from modules.train import train
from modules.evaluation import evaluated_model, evaluated_model_BLUE_only

from transformers import AutoModelForCausalLM


def main() -> None:
    """
    Main structure of the implementation
    """
    initialize_wandb_environment()  # initialize the wandb environmental variables
    args = parse_argument()  # parse the input arguments into a dictionary
    training_argument = get_training_argument(args)  # Parse the command line argument to trainer training argument
    model_dir = args.output_dir  # We create the directory name of the model based on the input arguments
    transformer_model, tokenizer = get_model_and_tokenizer(args, model_dir)  # Load the desired (base) Transformer model and its corresponding tokenizer
    train_dataset, val_dataset, test_dataset = load_dataset(args, tokenizer)

    trained_model = train(args, training_argument, transformer_model, tokenizer, train_dataset, val_dataset, model_dir)
    language_model = load_model_from_directory(AutoModelForCausalLM, "lm_model/")
    trained_model.to(args.device)  # assign the loaded model to the correct device
    language_model.to(args.device)
    language_model.resize_token_embeddings(len(tokenizer))

    giga_perplexity, BLUE_score, giga_generated_summary_length, giga_ave_time_consumption = \
        evaluated_model_BLUE_only(args, trained_model, test_dataset, tokenizer)

    print("Performance on Yuqiao dataset \n")
    print("BLUE-1: ", round(100 * BLUE_score[0], 2))
    print("BLUE-2: ", round(100 * BLUE_score[1], 2))
    print("BLUE-3: ", round(100 * BLUE_score[2], 2))
    print("BLUE-4: ", round(100 * BLUE_score[3], 2))
    print("Average (per sample) generated summary length", giga_generated_summary_length)
    print("Average (per sample) inference time consumption on the test dataset", giga_ave_time_consumption)

    print("Everything done: exiting the script...\n")
    print("Notice we don't save the result to local files, but it will be saved in wandb...\n")


if __name__ == '__main__':
    main()
