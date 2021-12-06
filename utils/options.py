# -----------------------------------------------------------
# Handle the customized training arguments
# (C) 2021 Puyuan Liu, Department of Computing Science, University of Alberta
# Released under GNU Public License (GPL)
# Email puyuan@ualberta.ca
# -----------------------------------------------------------

import torch
import argparse
from dataclasses import dataclass, field
from transformers import TrainingArguments
from copy import deepcopy

from distutils.util import strtobool


def parse_argument() -> argparse.Namespace:
    """
    The function to parse arguments entered in the command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", default=5e-5, type=float, required=False, help="learning rate")
    parser.add_argument("--seed", default=42, type=int, required=False, help="seed to replicate results")
    parser.add_argument("--gradient_accumulation_steps", default=32, type=int, required=False,
                        help="gradient_accumulation_steps")
    parser.add_argument("--batch_size", default=1, type=int, required=False, help="batch_size")
    parser.add_argument("--num_workers", default=0, type=int, required=False, help="num of cpus available")
    parser.add_argument("--device", default=torch.device('cuda'), required=False, help="torch.device object")
    parser.add_argument("--num_train_epochs", default=5, type=int, required=False, help="no of epochs of training")
    parser.add_argument("--output_dir", default="outputs/", type=str, required=False,
                        help="path to save evaluation results")
    parser.add_argument("--model_dir", default="weights/", type=str, required=False, help="path to save trained model")
    parser.add_argument("--fp16", default=True, type=bool, required=False,
                        help="whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="max gradient norm.")
    parser.add_argument("--train_file_dir", default='HC_training_data/', type=str,
                        help="location of training dataset.")
    parser.add_argument("--val_file_dir", default='HC_validation_data/', type=str,
                        help="location of validation dataset.")
    parser.add_argument("--test_file_dir", default='HC_test_data/Giga/', type=str,
                        help="location of giga test dataset.")
    parser.add_argument("--summary_length", default=8, type=int, help="desired summary length")
    parser.add_argument("--max_length", default=256, type=int,
                        help="maximum sentence length (article+summary in bytes)")
    parser.add_argument("--model_name", default="gpt2", type=str, help="the Transformer model for training")
    parser.add_argument("--do_train", default=True, type=strtobool, required=False,
                        help="Whether to run training or not.")
    parser.add_argument("--do_eval", default=True, type=strtobool, required=False,
                        help="Whether to run evaluation on the validation set or not.")

    parser.add_argument("--evaluation_strategy", default="epoch", type=str, required=False,
                        help="The evaluation strategy to adopt during training. (no, steps, epoch).")
    parser.add_argument("--load_best_model_at_end", default=True, type=strtobool, required=False,
                        help="Whether or not to load the best model found during training at the end of training.")
    parser.add_argument("--metric_for_best_model", default="loss", type=str, required=False,  # eval_loss
                        help="Must be the name of a metric returned by the evaluation with or without the prefix eval_.")
    parser.add_argument("--dataloader_drop_last", default=True, type=strtobool, required=False,
                        help="Whether to drop the last incomplete batch (if the length of the dataset is not divisible "
                             "by the batch size) or not.")
    parser.add_argument("--report_to", default="wandb", type=str, required=False,
                        help="Default wandb.")
    parser.add_argument("--logging_steps", default=1, type=int, required=False,
                        help="Frequency of logging metrics.")
    parser.add_argument("--validation_criteria", default="loss", type=str, required=False,
                        help="Whether to use rouge or loss as the validation criteria")
    parser.add_argument("--resume_training", default=False, type=strtobool, required=False,
                        help="Whether to resume train based on previously saved weight")
    parser.add_argument("--train_language_model", default=True, type=strtobool, required=False,
                        help="Whether do we train language model based on the summary dataset")
    args = parser.parse_args()
    return args


@dataclass
class CustomizedTrainingArguments(TrainingArguments):
    """
    Since we want the transformer trainer to automatically log everything to wandb while by default it's only logging
    the argument in the Trainingargument as the config for wandb, we created a customized training argument class
    such that our own argument can also be included in the wandb config.
    """
    giga_test_length:int = field(default=8)
    max_length:int =field(default=256)
    model_dir:str = field(default="/")
    test_file_dir:str = field(default="/")
    train_file_dir:str = field(default="/")
    val_file_dir:str = field(default="/")
    train_language_model:bool = field(default=False)
    model_name:str = field(default="gpt2")
    resume_training:bool = field(default=False)
    summary_length:int = field(default=False)
    device: str = field(default="cuda")
    validation_criteria: str = field(default="loss")


def get_training_argument(args: argparse.Namespace) -> CustomizedTrainingArguments:
    """
    The function to get the training argument for the trainer
    :param args: (dict) the command line arguments
    :param arg_type: (TrainingArguments) the type of the training argument
    :return: (TrainingArguments) the parsed version of the training argument
    """
    output_dir = "%s/%s_batch_%d_epochs_%d_summary_length_%d_max_length_%d" \
                 % (args.output_dir, args.model_name, args.batch_size, args.num_train_epochs,
                    args.summary_length, args.max_length)
    args.output_dir = output_dir
    args.per_device_train_batch_size = args.batch_size
    args.per_device_eval_batch_size = args.batch_size
    args.dataloader_num_workers = args.num_workers
    args.save_strategy = args.evaluation_strategy
    new_args = deepcopy(args)
    del new_args.batch_size
    del new_args.num_workers
    training_argument = CustomizedTrainingArguments(**vars(new_args))
    return training_argument


# Unit test
# my_argument = parse_argument()
# training_argument = get_training_argument(my_argument)
# print("passed")
