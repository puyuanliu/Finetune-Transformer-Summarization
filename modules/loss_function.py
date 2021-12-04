# -----------------------------------------------------------
# Loss function for different Transformers on summarization task
# (C) 2021 Puyuan Liu, Department of Computing Science, University of Alberta
# Released under GNU Public License (GPL)
# email puyuan@ualberta.ca
# -----------------------------------------------------------
import torch
import argparse
from types import FunctionType


def get_gpt2_loss_function(args: argparse.Namespace, tokenizer: any,loss_fct: FunctionType) -> FunctionType:
    """
    Return a function which calculates the loss when training GPT2 for summarization task
    """
    def gpt2_loss_function(self, model, inputs, return_outputs=False):
        """
        The function to calculate the loss for GPT2 when training on a summarization task.
        Though the model inference part has parallelism, the loss function requires further changing to enable parallelism
        """
        outputs = model(**inputs)  # Get the output by passing the current input to the model
        logits = outputs.logits  # Get the logits of outputs
        batch_loss = torch.zeros([args.batch_size], device=logits.device)  # We store the loss of every sample to a list then take average to get batch loss
        # Input and the labels are essentially the same since they have the structure
        # article + <|sep|> + summary
        # Also notice the model is making predictions based on previous ground truth token (teacher forcing)
        article = inputs["input_ids"]
        labels = inputs["input_ids"]
        for i in range(0, args.batch_size):
            # Get the idx of the <|sep|> token
            idx = (article[i] == tokenizer.sep_token_id).nonzero(as_tuple=False).item()
            # only consider loss on reference summary just like seq2seq models
            eos_token_pos = ((labels[i] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0])
            shift_logits = logits[i][..., idx:eos_token_pos - 2, :].contiguous()  # We only calculate loss for non-padding and non-terminating tokens.
            shift_labels = labels[i][..., idx + 1:eos_token_pos-1].contiguous()
            # Change the end of sentence token to padding token such that it will be ignored during loss calculation
            #shift_labels[shift_labels == tokenizer.eos_token_id] = tokenizer.pad_token_id
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))  # Calculate the loss
            loss = loss / args.gradient_accumulation_steps  # Using gradient_accumulation trick
            batch_loss[i] = loss
        batch_mean_loss = torch.mean(batch_loss)  # Calculate the batch loss by taking average
        return (batch_mean_loss, outputs) if return_outputs else batch_mean_loss
    return gpt2_loss_function

