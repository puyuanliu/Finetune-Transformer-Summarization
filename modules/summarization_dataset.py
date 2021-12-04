#!/usr/bin/env python

# -----------------------------------------------------------
# Dataset builder of gigaword & HC summary.
# This file serves as the helper file for the project "unsupervised sentence summarization"
# which is supervised by Dr. Mou (Lili Mou).
# (C) 2021 Puyuan Liu, Department of Computing Science, University of Alberta
# Released under GNU Public License (GPL)
# Email puyuan@ualberta.ca
# -----------------------------------------------------------

import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from my_errors import SelfConflictingError


class GPT2Dataset(Dataset):
    """
    The gigaword (with HC summary dataset) when using GPT2
    """

    def __init__(self, article, summaries, tokenizer, summary_length, data_size=None, smooth_prob=0,
                 max_length=128, smoothing_source=None):
        """
        Initialization of the dataset class
        Args:
            article: (string) The filename of the article.
            summaries: (string) The filename of the (hc) summary.
            tokenizer: (transformer tokenizer) The GPT2 tokenizer to perform encode and decode
            summary_length: (int) Desired summary length.
            data_size: (int) The number of samples we are going to use (needs to be less or equal
                                than the size of input article/summaries
            smooth_prob: (float) The probability of using sentences from smoothing source as training target
            max_length: (int) Maximum length of the concentrated
                        (decoded article + sep_token + decoded summary + end token) token list.
            smoothing_source: (str) The source that we ar going to perform smoothing {"article", "summary"}
            duc: (bool) Whether we are saving duc test dataset
        """
        self.max_length = max_length
        with open(article, "r") as f:
            # Read the article txt file from the given filename
            article = f.readlines()
        article = [sentence.rstrip().rstrip(" .") for sentence in tqdm(article)]  # remove stopping period
        # article = [sentence.rstrip() for sentence in tqdm(article)]  # temperately stop removing it since the previous trained model was trained with stopping period
        with open(summaries, "r") as f:
            # Read the summary txt file from the given filename
            summaries = f.readlines()
        summaries = [sentence.rstrip().rstrip(" .") for sentence in tqdm(summaries)]  # remove stopping period
        # summaries = [sentence.rstrip() for sentence in tqdm(summaries)]
        ref_summaries = summaries  # This become a dummy variable now
        self.article = article
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.ref_summaries = ref_summaries
        self.smooth_prob = smooth_prob
        self.smoothing_source = smoothing_source
        self.summary_length = summary_length
        self.end_sentence_token = tokenizer.bos_token
        self.separation_token = tokenizer.sep_token
        if data_size is None:
            # We didn't specify the size of the dataset, we use the maximum size of the given data
            self.len = len(self.article)
        else:
            # Otherwise we use the given length as the size of the dataset
            # Notice we didn't include error checking here
            self.len = data_size

    def __len__(self):
        # Return the size of the dataset
        return self.len

    def __getitem__(self, idx):
        # file_name = os.path.join(self.root_dir,str(idx)+".json")
        current_article = self.article[idx]  # Get the current article
        current_summary = self.summaries[idx]  # Get the current (hc) summary
        current_p = random.uniform(0, 1)  # Sampling the probability to perform smoothing
        if current_p <= self.smooth_prob:
            # If we are going to perform smoothing
            # We won't reach this part if the given smoothing probability is 0
            # We also only reach this part if we are doing training or validation
            if self.smoothing_source == "article":
                try:
                    # In case the current_article is too short to be clipped
                    concentration = self.clip_article(current_article)
                except:
                    # If it is indeed too short to be clipped, we ignore the smoothing
                    print("Current article is too short to be clipped...ignoring smoothing...\n")
                    # Concentrate the article and summary together
                    concentration = self.concentrate(current_article, current_summary)
            elif self.smoothing_source == "summary":
                # If we are going to clip some reference summaries as the alternative training target
                sampled_summary = self.sample_summary()  # Sample the summary from the reference summary dataset
                concentration = self.concentrate(current_article, sampled_summary)  # concentrate article and clipped reference summary
            elif self.smoothing_source is None:
                # If there is no smoothing source but there is still smoothing probability, there is a self conflicitng error
                raise SelfConflictingError(
                    "You cannot have non-zero smoothing probability with out a smoothing source.\n")
            else:
                raise ValueError(
                    "Unrecognized smoothing source, the valid smoothing source are {article, summary, None}. \n")
        else:
            # Otherwise we simply concentrate the article and the summary
            concentration = self.concentrate(current_article, current_summary)
        # Encoded the concentration and pad it to our desired length
        encoded_concentration = self.tokenizer.encode(concentration, padding="max_length", max_length=self.max_length)
        encoded_concentration = torch.tensor(encoded_concentration)  # Concentrate tensors together
        return {"input_ids": encoded_concentration, "labels": encoded_concentration}

    def concentrate(self, current_article, current_summary):
        """
        Concentrate the article and summary, and add special tokens, then encode it
        :param article: a article string of the sample
        :param summary: a summary string of the sample
        :return: the concentrated encoded version of the article+string
        """
        concentration = self.join_pieces(current_article, current_summary)  # Concentrate article and summary
        encoded_concentration = self.tokenizer.encode(concentration)  # Encode the concentration
        if len(encoded_concentration) > self.max_length:
            # If the length of the joint sentence (article + summary) is longer than the maximum length
            # we sample another sentence as the replacement
            print("Length Exceeded, sampling... \n")
            while True:
                # We keep sampling summaries until finding a summary that fits the max length criteria
                current_idx = np.random.randint(0, self.len)  # Get the sampling index
                current_article = self.article[current_idx]  # Sample the current article
                current_summary = self.summaries[current_idx]  # Sample the current summary
                concentration = self.join_pieces(current_article, current_summary)  # Concentrate the new article and summary
                encoded_concentration = self.tokenizer.encode(concentration)  # Encode the concentration
                if len(encoded_concentration) <= self.max_length:
                    # If the new encoded concentration has a desired length
                    # We break the loop
                    break
        # Notice we don't return the encoded concentration
        # The encoded concentration was only used to test whether the concentration wil be greateer than desired length.
        return concentration

    def clip_article(self, current_article):
        """
        concentrate the input article and a clipped version of the article
        Args:
            current_article: (str) the current article that we are working on.
        Returns: (str) Concentration of article and clipped article.

        """
        current_article_list = current_article.split(" ")  # Split the string into a list of tokens
        # Randomly sampling a starting point to perform clipping
        starting_point = random.randrange(0, len(current_article_list) - self.summary_length)
        # Get a string of the clipped article
        clipped_article = " ".join(current_article_list[starting_point:starting_point + self.summary_length])
        # Join the original article and the new article together
        concentration = self.join_pieces(current_article, clipped_article)
        # Encode the concentration
        encoded_concentration = self.tokenizer.encode(concentration)
        if len(encoded_concentration) <= self.max_length:
            # If the encoded concentration does not break the max length constraint, we return the concentration
            return concentration
        else:
            # Otherwise, we raise a error
            # (This error should be caught by parent functions)
            print("The length of the encoded sentence concentration is greater than the maximum tolerable length\n")
            raise SelfConflictingError(
                "The length of the encoded sentence concentration is greater than the maximum tolerable length")

    def sample_summary(self):
        """
        Sample a summary that is greater than or equal to the desired length from the reference summary dataset
        :return:
        """
        current_idx = np.random.randint(0, self.len)  # Randomly sampling a reference summary index
        current_summary = self.ref_summaries[current_idx]  # Get the string of the reference summary
        current_summary_list = current_summary.split(" ")  # Split the string into a list of tokens
        if len(current_summary_list) > self.summary_length:
            # If the generated summary have a larger length than desired
            # We sample a starting point to clip
            starting_point = random.randrange(0, len(current_summary_list) - self.summary_length)
            # get a string of the clipped sample summary
            current_summary = " ".join(current_summary_list[starting_point:starting_point + self.summary_length])
        # Notice we don't bother re-sampling if the summary length is shorter than desired
        # Later we will use decoder to pad all (token) concentrations to have the same length
        return current_summary

    def join_pieces(self, sentence_1, sentence_2):
        """
        Joint two sentence together with the separate token from the tokenizer
        :param sentence_1: the first sentence (e.g., article)
        :param sentence_2: the second sentence (e.g., summary)
        :return: joint sentence
        """
        # GPT2 treat " token" and "token" as different token, and tokens without leading space was usually
        # the case of being the starting token to a sentence. Since we want a complete sentence (summary)
        # to be generated following the <|sep|> token, we remove the leading space of the first token
        # following the <|sep|> token.
        # Updated: we add an extra space in front of the end_sentence_token to help later rouge score calculation
        # (Since we want to separate the string according to the spacing during generation of reference summaries)
        return (sentence_1 + self.separation_token + sentence_2 + " " + self.end_sentence_token).replace("<|unk|>", "UNK")
