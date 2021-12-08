# -----------------------------------------------------------
# Handle the customized training arguments
# (C) 2021 Puyuan Liu, Department of Computing Science, University of Alberta
# Released under GNU Public License (GPL)
# Email puyuan@ualberta.ca
# -----------------------------------------------------------
from tqdm import tqdm
import argparse
import numpy as np
from copy import deepcopy
from types import FunctionType
from typing import Tuple, List
from pythonrouge.pythonrouge import Pythonrouge

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from utils.helper import save_summary
from modules.inference import sample_seq_text
from nltk.translate.bleu_score import corpus_bleu as BLUE


def get_gpt2_evaluation_function(args: argparse, tokenizer: any, loss_fct:FunctionType):
    def gpt2_eval_metric(prediction: Tensor):
        """
        This function evaluates the predicted result returned by the trainer
        """
        max_prob_predictions = prediction[0]
        labels = prediction[1]
        generated_summaries = []  # Initialize the summary and reference list to be empty
        ref_summaries = []
        for i in range(0, len(max_prob_predictions)):
            sep_index = (labels[i] == tokenizer.sep_token_id).nonzero()[0][0]  # Get the index of the separate token
            # clipping the logits to only include the generated summary part
            # If the model determined eos token
            if len((max_prob_predictions[i][sep_index:] == tokenizer.eos_token_id).nonzero()[0]) != 0:
                generated_summary_end_position = sep_index + (max_prob_predictions[i][sep_index:]
                                                              == tokenizer.eos_token_id).nonzero()[0][0]
            elif len((max_prob_predictions[i][sep_index:] == tokenizer.sep_token_id).nonzero()[0]) != 0:
                generated_summary_end_position = sep_index + (max_prob_predictions[i][sep_index:]
                                                              == tokenizer.sep_token_id).nonzero()[0][0]
            else:
                generated_summary_end_position = sep_index + 10
            current_generated_summary = tokenizer.decode(max_prob_predictions[i][sep_index:generated_summary_end_position]) # We clipped the generated summary by desired length
            ref_summary_end_position = (labels[i] == tokenizer.eos_token_id).nonzero()[0][0]
            current_ref_summary = tokenizer.decode(
                labels[i][sep_index + 1:ref_summary_end_position])  # decode the reference tokens to ref summary
            generated_summaries.append(
                ''.join([i if ord(i) < 128 else ' ' for i in current_generated_summary]))  # remove non-gdb(ascii) token
            # generated_summaries.append(current_generated_summary)
            ref_summaries.append(current_ref_summary)
        if args.validation_criteria == "rouge":
            # If we want to use the rouge score as the validation criteria
            rouge = Pythonrouge(summary_file_exist=False,
                                summary=[[i] for i in generated_summaries], reference=[[[i]] for i in ref_summaries],
                                n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                                stemming=True, stopwords=False,
                                word_level=False, length_limit=False, length=100,
                                use_cf=True, cf=95, scoring_formula='average',
                                resampling=True, samples=1000, favor=True, p=0.5)
            print("Start calculating rouge score...")
            scores = rouge.calc_score()  # Calculate the rouge score
            # scores = {"ROUGE-1-F":10, "ROUGE-2-F":20, "ROUGE-L-F":15 }
            scores["ROUGE-1-F"] *= 100  # Rescale the rouge score to 100 times larger
            scores["ROUGE-2-F"] *= 100
            scores["ROUGE-L-F"] *= 100
            rouge_geometric = np.log(np.array([scores["ROUGE-1-F"], scores["ROUGE-2-F"], scores["ROUGE-L-F"]]))
            rouge_geometric = np.exp(rouge_geometric.mean())
            # Return the a metric directory, though rouge score is also included, we tend to use cross-entropy loss to
            # perform validation (model selection)
            eval_result = {"ROUGE-1-F": scores["ROUGE-1-F"], "ROUGE-2-F": scores["ROUGE-2-F"],
                                "ROUGE-L-F": scores["ROUGE-L-F"], "ROUGE-GEO": rouge_geometric}

        elif args.validation_criteria == "BLUE-4":
            hypothesis = [i.split() for i in generated_summaries]
            references = [[i.split()] for i in ref_summaries]
            BLUE_4 = BLUE(references, hypothesis, weights=(0, 0, 0, 1))
            eval_result = {"BLUE-4": BLUE_4}

        else:
            # If we want to use cross-entropy loss as the validation criteria
            eval_result = {}
        return eval_result

    def gpt2_prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):

        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        inputs = self._prepare_inputs(inputs)
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss.mean().detach()
        if isinstance(outputs, dict):
            logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
        else:
            logits = outputs[1:]
        if len(logits) == 1:
            logits = logits[0]
        max_prob_predictions = logits.argmax(axis=2)
        return (loss, max_prob_predictions, inputs["labels"])

    return [gpt2_eval_metric, gpt2_prediction_step]


def evaluated_model(args, model, test_dataset, tokenizer, use_finetune=True, unfintuned_model=None,
                    n_iter=10, summary_save_directory="generated_summaries", language_model=None):
    """
    Evaluate the given model on the given test dataset and save the generated summaries
    """

    model.eval()  # Set the model to evaluation mode
    rouge_score_r1_list = []  # Initialize lists to store rouge score for each iteration such that we can calculate their average later
    rouge_score_r2_list = []
    rouge_score_rl_list = []
    perplexity_list = []   # Initialize the list to store perplexity for each iteartion
    length_list = []  # Initialize list to store the average summary length of the generated summary of each iteration
    time_consumption_list = []
    ignore_index = tokenizer.pad_token_id
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index)  # Ignore padding index when calculating cross-entropy loss
    for i in tqdm(range(0, n_iter)):
        # evaluate the rouge score of the given model on the given dataset
        # we also include the time consumption of model inference and the output
        scores, generated_summary, time_consumption = evaluate_rouge_score(args, model, test_dataset, tokenizer, num=i)
        time_consumption_list.append(time_consumption)
        rouge_score_r1_list.append(scores["ROUGE-1-F"])
        rouge_score_r2_list.append(scores["ROUGE-2-F"])
        rouge_score_rl_list.append(scores["ROUGE-L-F"])
        save_summary(args, generated_summary, summary_save_directory, i)  # save the generated summary to directory
        # Calculate the average length of the generated summary
        # Notice we remove the dot token as what was done in previous works (e.g., HC_summary)
        current_ave_length_list = [len(sentence[0].split()) for sentence in generated_summary]
        current_ave_length = sum(current_ave_length_list)/len(current_ave_length_list)
        length_list.append(current_ave_length)
        if use_finetune:
            # If we use the finetuned the model to calculate the perplexity, we assign the perp_model to be the current trained model
            perp_model = language_model
        else:
            # If we use the unfinetuned the model to calculate the perplexity, we assign the perp_model to be the unfinetuned model
            perp_model = unfintuned_model
        current_perplexity = evaluate_perplexity(args, perp_model, tokenizer, ignore_index, generated_summary, loss_fct)
        perplexity_list.append(current_perplexity)
    # Average the metrics over iterations.
    ave_perplexity = sum(perplexity_list) / len(perplexity_list)
    ave_rouge_score = [sum(rouge_score_r1_list) / len(rouge_score_r1_list),
                       sum(rouge_score_r2_list) / len(rouge_score_r2_list)
        , sum(rouge_score_rl_list) / len(rouge_score_rl_list)]
    ave_summary_length = sum(length_list)/len(length_list)
    ave_time_consumption = sum(time_consumption_list)/len(time_consumption_list)
    return ave_perplexity, ave_rouge_score, ave_summary_length, ave_time_consumption


def evaluated_model_BLUE_only(args, model, test_dataset, tokenizer, n_iter=10,
                              summary_save_directory="generated_summaries"):
    model.eval()  # Set the model to evaluation mode
    BLUE_1_list = []  # Initialize lists to store rouge score for each iteration such that we can calculate their average later
    BLUE_2_list = []
    BLUE_3_list = []
    BLUE_4_list = []
    length_list = []  # Initialize list to store the average summary length of the generated summary of each iteration
    time_consumption_list = []
    ignore_index = tokenizer.pad_token_id
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index)  # Ignore padding index when calculating cross-entropy loss
    for i in tqdm(range(0, n_iter)):
        # evaluate the rouge score of the given model on the given dataset
        # we also include the time consumption of model inference and the output
        scores, generated_summary, time_consumption = evaluate_BLUE_score(args, model, test_dataset, tokenizer, num=i)
        time_consumption_list.append(time_consumption)
        BLUE_1_list.append(scores[0])
        BLUE_2_list.append(scores[1])
        BLUE_3_list.append(scores[2])
        BLUE_4_list.append(scores[3])
        save_summary(args, generated_summary, summary_save_directory, i)  # save the generated summary to directory
        # Calculate the average length of the generated summary
        # Notice we remove the dot token as what was done in previous works (e.g., HC_summary)
        current_ave_length_list = [len(sentence[0].split()) for sentence in generated_summary]
        current_ave_length = sum(current_ave_length_list)/len(current_ave_length_list)
        length_list.append(current_ave_length)

    # Average the metrics over iterations.
    ave_BLUE_score = [sum(BLUE_1_list) / len(BLUE_1_list), sum(BLUE_2_list) / len(BLUE_2_list),
                      sum(BLUE_3_list) / len(BLUE_3_list), sum(BLUE_4_list) / len(BLUE_4_list)]
    ave_summary_length = sum(length_list)/len(length_list)
    ave_time_consumption = sum(time_consumption_list)/len(time_consumption_list)
    return None, ave_BLUE_score, ave_summary_length, ave_time_consumption


def evaluate_perplexity(args: argparse.Namespace, perp_model: any, tokenizer: any, ignore_index: List[int], generated_summary: List[str], loss_fct):
    """
    Evaluate the perplexity of the generated summaries given the perplexity model
    """
    eval_loss = 0.0  # Initialize the evaluation loss to be 0 such that we get the total loss to calculate the averange at the end
    num_steps = 0

    processed_generated_summary = deepcopy(generated_summary)
    for i in reversed(range(0, len(generated_summary))):
        processed_generated_summary[i] = [generated_summary[i][0]]
        if processed_generated_summary[i][0][:7] == "UNK UNK" or processed_generated_summary[i][0][:6] == "UNKUNK":
            del processed_generated_summary[i]
    encoded_generated_summary = [torch.tensor(tokenizer.encode(i[0], padding="max_length", max_length=64)) for i in
                                 processed_generated_summary]  # Encode every sentence in the generated summary and store as a list
    encoded_generated_summary = torch.stack(encoded_generated_summary,
                                            dim=0)  # Stack the list to a n (number sentences) tensor
    encoded_generated_summary_dataloader = DataLoader(encoded_generated_summary, batch_size=args.batch_size,
                                                      drop_last=True)  # Create a dataloader such that we can draw batches efficently
    print("Start Calculating Perplexity")
    for encoded_generated_summary in tqdm(encoded_generated_summary_dataloader):
        encoded_generated_summary = encoded_generated_summary.to(
            args.device)  # Move the current batch to the same device as the model
        logits = perp_model(encoded_generated_summary)[0]  # Get the inference result logits
        # Calculate the cross-entropy loss by comparing the output logits and the desired labels.
        lm_loss = loss_fct(logits[:, 0:-1].clone().view(-1, logits.size(-1)), encoded_generated_summary[:, 1:].clone().view(-1))
        eval_loss += lm_loss.item()  # Add the loss to the total evaluation loss
        num_steps += 1
    # Calculate the perplexity by taking exponential of the average cross-entropy loss
    return np.exp(eval_loss / num_steps)


def evaluate_rouge_score(args: argparse.Namespace, model: any, test_dataset:any, tokenizer: any, temperature=0.8,
                         top_k=10, top_p=0.5, num=0) -> Tuple:
    """
    Evaluate the rouge score given the model and test dataset.
    """
    generated_summary = []  # initialize the generated summary and reference summary list
    reference_summary = []
    total_time_consumption = []
    for i in tqdm(range(0, len(test_dataset))):
        current_sample = test_dataset[i]  # get the i-th concentrated tokens from the dataset, note this is a dict
        current_ref_summary = test_dataset.ref_summaries[i]  # get the i-th reference summary from the dataset, note this is str
        current_sample = current_sample["input_ids"]  # get the concentrated token list from the dict
        idx = (current_sample == tokenizer.sep_token_id).nonzero(as_tuple=False).item()  # get the index of the sep token
        current_context = current_sample[:idx + 1].tolist()  # get tokens before the sep token (which is the article part)
        # get the summary with the desired length
        current_generated_summary, sample_time_consumption = sample_seq_text(model, current_context, args.summary_length,
                                                    args.device, temperature, top_k, top_p,
                                                    tokenizer=tokenizer)
        total_time_consumption.append(sample_time_consumption)
        generated_summary.append([''.join([i if ord(i) < 128 else ' ' for i in current_generated_summary])])  # We remove the stopping period from the generated summaries
        reference_summary.append([[current_ref_summary]])

    rouge = Pythonrouge(summary_file_exist=False,
                        summary=generated_summary, reference=reference_summary,
                        n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                        stemming=True, stopwords=False,
                        word_level=False, length_limit=False, length=100,
                        use_cf=True, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)

    print("Start calculating rouge score...")
    scores = rouge.calc_score()
    return scores, generated_summary, sum(total_time_consumption)/len(total_time_consumption)


def evaluate_BLUE_score(args: argparse.Namespace, model: any, test_dataset:any, tokenizer: any, temperature=0.8,
                         top_k=10, top_p=0.5, num=0) -> Tuple:
    """
    Evaluate the rouge score given the model and test dataset.
    """
    generated_summary = []  # initialize the generated summary and reference summary list
    reference_summary = []
    total_time_consumption = []
    for i in tqdm(range(0, len(test_dataset))):
        current_sample = test_dataset[i]  # get the i-th concentrated tokens from the dataset, note this is a dict
        current_ref_summary = test_dataset.ref_summaries[i]  # get the i-th reference summary from the dataset, note this is str
        current_sample = current_sample["input_ids"]  # get the concentrated token list from the dict
        idx = (current_sample == tokenizer.sep_token_id).nonzero(as_tuple=False).item()  # get the index of the sep token
        current_context = current_sample[:idx + 1].tolist()  # get tokens before the sep token (which is the article part)
        # get the summary with the desired length
        current_generated_summary, sample_time_consumption = sample_seq_text(model, current_context, args.summary_length,
                                                    args.device, temperature, top_k, top_p,
                                                    tokenizer=tokenizer)
        total_time_consumption.append(sample_time_consumption)
        generated_summary.append(''.join([i if ord(i) < 128 else ' ' for i in current_generated_summary]))  # We remove the stopping period from the generated summaries
        reference_summary.append(current_ref_summary)

    hypothesis = [i.split() for i in generated_summary]
    references = [[i.split()] for i in reference_summary]
    scores = []
    for i in range(1, 5):
        scores.append(BLUE(references, hypothesis, weights=(1,)*i))
    return scores, generated_summary, sum(total_time_consumption)/len(total_time_consumption)


