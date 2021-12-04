# -----------------------------------------------------------
# Fine-tune (Transformer) models on summarization task
# (C) 2021 Puyuan Liu, Department of Computing Science, University of Alberta
# Released under GNU Public License (GPL)
# Email puyuan@ualberta.ca
# -----------------------------------------------------------
import argparse
import shutil
import types
from utils.helper import load_model_from_directory, save_model
from transformers import AutoModelForCausalLM, DataCollatorWithPadding, Trainer
from utils.options import CustomizedTrainingArguments
from utils.load_data import GPT2Dataset
from torch.nn import CrossEntropyLoss
from modules.loss_function import get_gpt2_loss_function
from modules.evaluation import get_gpt2_evaluation_function


def train(args: argparse.Namespace, training_argument:CustomizedTrainingArguments, transformer_model: any,
          tokenizer: any, train_dataset: GPT2Dataset, val_dataset: GPT2Dataset, model_dir: str):
    """
    Train the adopted Transformer
    """
    if args.do_train:
        # If we are going to train
        if "gpt2" in args.model_name: # If the adopted Transformer is gpt2 or its variant
            trained_model = train_gpt2(transformer_model, tokenizer, args, training_argument, train_dataset, val_dataset, model_dir)
        else:
            raise NotImplementedError
        print("Loading trained model...\n")
    else:
        # Load the model saved during the training
        trained_model = load_model_from_directory(AutoModelForCausalLM, model_dir)
    return trained_model


def train_gpt2(model: any, tokenizer: any, args: argparse.Namespace, training_argument: CustomizedTrainingArguments,
               training_dataset: GPT2Dataset, validation_dataset: GPT2Dataset, model_dir: str):
    """
    Perform the training of GPT2 kind model
    """
    print("Start training...\n")
    # resize the embedding layer of the model, otherwise cuda error occurs :(, sadly this is not auto-done by trainer
    ignore_index = tokenizer.pad_token_id
    # we use a customized GPT2 loss function since the model was not designed for summarization task at the beginning
    # and requires revising on the loss function
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
    # Create the loss function for GPT2 training
    gpt2_loss_function = get_gpt2_loss_function(args, tokenizer, loss_fct)
    evaluation_functions = get_gpt2_evaluation_function(args, tokenizer, loss_fct)  # Create the evaluation function for GPT2 training
    gpt2_eval_metric, gpt2_prediction_step = evaluation_functions[0], evaluation_functions[1]
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=args.max_length)
    model_trainer = Trainer(model, training_argument, train_dataset=training_dataset, eval_dataset=validation_dataset,
                            compute_metrics=gpt2_eval_metric, data_collator=data_collator)  # Customize the trainer
    # Set the loss calculation function to our customized loss function
    model_trainer.compute_loss = types.MethodType(gpt2_loss_function, model_trainer)
    # Set the inference function to our customized inference function
    model_trainer.prediction_step = types.MethodType(gpt2_prediction_step, model_trainer)
    model_trainer.train()  # Perform the actual training
    shutil.rmtree(model_dir)  # Remove the checkpoints otherwise we will run out of the github storage...
    save_model(model, model_dir)

    return model
