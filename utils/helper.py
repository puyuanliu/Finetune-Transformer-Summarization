# -----------------------------------------------------------
# Handle the customized training arguments
# (C) 2021 Puyuan Liu, Department of Computing Science, University of Alberta
# Released under GNU Public License (GPL)
# Email puyuan@ualberta.ca
# -----------------------------------------------------------
import argparse
import os
from torch import TensorType
import torch.nn.functional as F
from typing import List
from pathlib import Path
import torch


def initialize_wandb_environment() -> None:
    """
    Initialize the WANDB environmental variable.
    """
    os.environ['WANDB_LOG_MODEL'] = "true"  # Logging the model to W&B Artifacts
    os.environ['WANDB_PROJECT'] = "unsupevised_finetune_transformer_summarization"  # Set the name of the project
    os.environ['WANDB_WATCH'] = "all"  # Log histograms of gradients and parameters
    # WANDB_DISABLED  Set to true to disable logging entirely (false by default)
    # WANDB_SILENT   Set to true to silence the output printed by wandb (false by default)


def load_model_from_directory(base_model: any, model_dir: str) -> any:
    """
    Load the model weights from file
    """
    model = base_model.from_pretrained(model_dir)  # load the saved model in the given directory
    return model


def save_model(model: any, model_dir: str) -> None:
    """
    Save the given model to given filenames
    """
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    model.save_pretrained(model_dir)


def load_trained_summaries(args: argparse.Namespace, base_summary_dir: str, i: int) -> List[List[str]]:
    """
    Load the previously saved summaries to evaluate at other metrics.
    """
    summary_filename = "%s/length%d/model_%s_trained_after_%d_epochs/%d.txt" \
                       % (base_summary_dir, args.summary_length, args.model_name, args.num_train_epochs, i)
    with open(summary_filename, "r") as f:
        generated_summaries = f.readlines()
        generated_summaries = [[sentence.rstrip()] for sentence in generated_summaries]
    return generated_summaries


def load_hc_summaries(length=8, hc_summary_base_folder="HC_summaries") -> List[List[str]]:
    """
    Load the HC summary
    """
    with open(hc_summary_base_folder + "/hc_title_%d/summaries.txt" % length, "r") as f:
        hc_summaries = f.readlines()
        hc_summaries = [[sentence.rstrip()] for sentence in hc_summaries]
    return hc_summaries


def generate_model_dir(args: argparse.Namespace) -> str:
    """
    This function generates the dir of our model
    """
    model_dir = "%s/%s_batch_%d_epochs_%d_summary_length_%d_max_length_%d" \
                % (args.output_dir, args.model_name, args.batch_size, args.num_train_epochs,
                   args.summary_length,  args.max_length)

    return model_dir


def save_summary(args: argparse.Namespace, generated_summary: List[str], base_summary_dir: str, i: int) -> None:
    """
    Save the generated summary into file
    """
    summary_dir = "%s/length%d/model_%s_trained_after_%d_epochs" \
                  % (base_summary_dir, args.summary_length, args.model_name,args.num_train_epochs)
    summary_filename = "%s/length%d/model_%s_trained_after_%d_epochs/%d.txt" \
                       % (base_summary_dir, args.summary_length, args.model_name, args.num_train_epochs, i)
    Path(summary_dir).mkdir(parents=True, exist_ok=True)
    print("Start writing generated summary to file %s \n" % summary_filename)
    with open(summary_filename, "w+") as f:
        for item in generated_summary:
            f.write("%s\n" % item[0])  # generated_summary by design have a structure [[sentence 1], ...[sentence n]]
    print("Finished writing!\n ")


def top_k_top_p_filtering(logits: TensorType, top_k=0, top_p=0.0, filter_value=-float('Inf')) -> TensorType:
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits
