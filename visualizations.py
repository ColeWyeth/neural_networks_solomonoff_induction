# %%
import functools
from typing import Any

from absl import app
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import tree

import pickle
import matplotlib.pyplot as plt

import sys
# import os
# sys.path.insert(0, os.path.abspath(".."))
nnsi_path = "/root/neural_networks_solomonoff_induction/.."
if f"{nnsi_path}" not in sys.path:
    sys.path.insert(0,f"{nnsi_path}")

from neural_networks_solomonoff_induction.data import data_generator as dg_lib
from neural_networks_solomonoff_induction.data import utm_data_generator as utm_dg_lib
from neural_networks_solomonoff_induction.data import utms as utms_lib
from neural_networks_solomonoff_induction.models import transformer

alphabet_size = 2
probabilistic = True
length = 256

generate_fresh_batch = True
# visualize = True

rng = np.random.default_rng(seed=8)

# %%
if generate_fresh_batch:
    program_sampler = utms_lib.MCSampler(
        rng=rng,
        filename='data/ctx_filtered_binary_probabilistic.pyd', # possibly data/ is necessary
    )
    # for i in range(10):
    #     p = program_sampler.sample_program(length)
    #     print(p)

    utm = utms_lib.BrainPhoqueUTM(
        program_sampler,
        alphabet_size = alphabet_size,
        print_trace = False,
        shorten_program = True,
        use_input_instruction = probabilistic,
    )
    data_generator = utm_dg_lib.UTMDataGenerator(
        batch_size=500,
        seq_length=256,
        rng=rng,
        utm=utm,
        memory_size=200, # matches optimize_Q
        maximum_steps=1024, # matches optimize_Q
        tokenizer=utm_dg_lib.Tokenizer.SEQ_POSITION,
        maximum_program_length=256,
    )

    batch, log_dict = data_generator.sample()
    batch = np.array(batch)

    print(batch.shape)
    binary_batch = batch.argmax(axis=-1)
    print(binary_batch.shape)
    #print(f"{binary_batch=}")
    for row in binary_batch:
        print(row)
    print(log_dict)

    with open('binary_batch_log_dict.pkl', 'wb') as f:
        pickle.dump((binary_batch,log_dict),f)

with open('binary_batch_log_dict.pkl', 'rb') as f:
    binary_batch, log_dict = pickle.load(f)

# %%
if False:
    config = transformer.TransformerConfig(vocab_size=alphabet_size)
    model = hk.transform(
        functools.partial(transformer.transformer_decoder, config=config)
    )
    fname = "params_mid_train.npz"
    with open(fname, "rb") as f:
        loaded_params = np.load(f, allow_pickle=True)
        params = dict(loaded_params)
        for k in params.keys():
            params[k] = params[k].item()
    conditionals = model.apply(
        params=params,
        targets=binary_batch,
        rng=None,
    )
    print(f"{conditionals.shape=}")
    true_conditionals = jnp.take_along_axis(
        conditionals, binary_batch[..., None], axis=-1
    )[..., 0]
    true_conditionals = jnp.where(log_dict['loss_mask'], 0.0, true_conditionals)
    print(true_conditionals)
    cond_probs = jax.numpy.exp(true_conditionals) # don't take seriously, check mask

    avg_log_loss = -true_conditionals.mean(axis=0)
    avg_cond_prob = cond_probs.mean(0)
    print(f"{avg_log_loss=}")

    print(f"{avg_cond_prob=}")

    total_loss = -true_conditionals.sum(axis=1)
    print(f"{total_loss=}")
    print([res['short_ln_loss'] for res in log_dict['results']])
    plt.plot(avg_log_loss)
    # plt.plot(avg_cond_prob)

# %%
# Spin up model
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

#torch.cuda.empty_cache()

# Load GPT-2 and tokenizer
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name,add_bos_token=True)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up SIT
config = transformer.TransformerConfig(vocab_size=alphabet_size)
model_sit = hk.transform(
    functools.partial(transformer.transformer_decoder, config=config)
)
fname = "params.npz"
with open(fname, "rb") as f:
    loaded_params = np.load(f, allow_pickle=True)
    params = dict(loaded_params)
    for k in params.keys():
        params[k] = params[k].item()


binary_ex = binary_batch[0][None,...]
total_log_probs_gpt = torch.zeros_like(torch.tensor(binary_ex))
total_log_probs_sit = np.zeros(binary_ex.shape)
examples_at_seq_pos = np.zeros(binary_ex.shape)

# %% 
def get_bit_log_probs(logits):
    # zero and one logits
    batch, seq_len, toks = logits.shape
    print(f"{logits.shape=}")
    bit_logits = torch.index_select(
        logits,
        1,
        2* torch.arange(seq_len//2).to(device),
    ) 
    print(f"{bit_logits.shape=}")
    # Note that for gpt2 tokenizer, 0->15, 1->16
    projected_logits = bit_logits[...,15:17]
    renormalized_log_probs = torch.nn.functional.log_softmax(projected_logits, dim=-1)
    return renormalized_log_probs

# %%
for i in range(binary_batch.shape[0]):
    binary_ex = binary_batch[i][None,...]
    real_seq = np.array(1 - log_dict['loss_mask'][i])
    print(binary_ex)
    # Prompt design 

    # string samples
    prompt = "" #"The following binary sequence was generated by a simple program, represented as comma-separated bits of output: "

    digit_names = {0: "zero", 1: "one"}

    #binary_batch = [256*[0]]

    str_ex = [
        prompt + ','.join([str(digit) for digit in sample]) for sample in binary_ex 
    ]
    #str_batch = [80*"This is a very repetitive sequence. ",]

    print(str_ex[0])
    print(len(str_ex))

    # Tokenize input
    inputs = tokenizer(str_ex, return_tensors="pt")
    print(f"{inputs=}")
    input_ids = inputs["input_ids"].to(device)

    # Run model
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits

    bit_log_probs = get_bit_log_probs(logits)

    # Select true surprisals
    true_bit_log_probs = bit_log_probs.gather(2, torch.tensor(binary_ex).to(device)[...,None]).squeeze(-1)
    print(f"{true_bit_log_probs.shape=}")

    # SIT
    conditionals = model_sit.apply(
        params=params,
        targets=binary_ex,
        rng=None,
    )
    print(f"{conditionals.shape=}")
    true_conditionals = jnp.take_along_axis(
        conditionals, binary_ex[..., None], axis=-1
    )[..., 0]

    total_log_probs_sit += np.array(true_conditionals)
    total_log_probs_gpt += true_bit_log_probs.cpu().numpy() * real_seq
    examples_at_seq_pos += real_seq

# %%
# Get ALL log probs
if False:
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Gather log probabilities of the correct tokens
    # [batch_size, seq_len, vocab_size] -> [batch_size, seq_len]
    token_log_probs = log_probs[:, :-1].gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    # Decode tokens and print with log probs
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    for token, log_prob in zip(tokens, token_log_probs[0]):
        print(f"{token:>12} : {log_prob.item():.4f}")


# %%
# Plotting
print(f"{total_log_probs_gpt.shape=}")
surprisal_gpt = -(total_log_probs_gpt[0].cpu().numpy() / examples_at_seq_pos[0])
print(surprisal_gpt.shape)
#print(surprisal)
cum_surprisal_gpt = surprisal_gpt.cumsum(axis=-1)
print(f"{cum_surprisal_gpt.shape=}")

surprisal_sit = -(total_log_probs_sit[0] / examples_at_seq_pos[0])
cum_surprisal_sit = surprisal_sit.cumsum(axis=-1)

plt.plot(cum_surprisal_gpt,label='gpt')
plt.plot(cum_surprisal_sit,label='sit')
plt.legend()

plt.title(f'{model_name} surprise on samples')
plt.xlabel('Token position')
plt.ylabel('Cumulative ln loss')
print([res['short_ln_loss'] for res in log_dict['results']])

# %%

