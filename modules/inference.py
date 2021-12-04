# -----------------------------------------------------------
# Predicts summaries given input document
# (C) 2021 Puyuan Liu, Department of Computing Science, University of Alberta
# Released under GNU Public License (GPL)
# email puyuan@ualberta.ca
# -----------------------------------------------------------

import time
import torch
from utils.helper import top_k_top_p_filtering
import torch.nn.functional as F


def sample_seq_text(model, context, length, device, temperature=1.0, top_k=0, top_p=0.0, tokenizer=None):
    """ Generates a sequence of tokens, notice this script does not support parallelism for now due to time constraint
        Since it's only used on the test dataset, it won't cause too many trouble.
        Notice this function is copied from the training script in https://github.com/SKRohit/Generating_Text_Summary_With_GPT2
        Args:
            model: (AutoModelForCausalLM) gpt/gpt2 model
            context: (List) tokenized text using gpt/gpt2 tokenizer
            length: (int) length of generated sequence.
            device: (torch.device) which device that we are going to perform operation on.
            temperature: (int>0) used to control the randomness of predictions by scaling the logits before applying softmax.
            top_k: (int>0)keep only top k tokens with highest probability (top-k filtering).
            top_p: (float>0.0) keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    context_length = len(context)
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    counter = 0
    sample_time_consumption = 0
    encoded_context_words = []
    decoded_context_split = tokenizer.decode(context[0,:-1].tolist()).split()
    for i in range(0,len(decoded_context_split)):
        current_code = tokenizer.encode(decoded_context_split[i])
        for code in current_code:
            encoded_context_words.append(code)
    with torch.no_grad():
        while counter < 2 * length:
            good_to_go = False
            inputs = {'input_ids': generated}
            start_time = time.time()
            outputs = model(
                **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            end_time = time.time()
            sample_time_consumption += end_time - start_time  # Calculate the time consumption of the model inference
            while not good_to_go:
                next_token_logits = outputs[0][0, -1, :] / temperature
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if next_token in context or next_token in encoded_context_words:
                    good_to_go = True
                else:
                    outputs[0][0, -1, next_token] = float("-inf")  # Remove the probability of that token
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            counter += 1
    generated_text = generated[0, context_length:].tolist()
    text = tokenizer.convert_ids_to_tokens(generated_text, skip_special_tokens=True)
    text = tokenizer.convert_tokens_to_string(text)
    text = text.split()[0:length]
    text = ' '.join(text)
    return text, sample_time_consumption
