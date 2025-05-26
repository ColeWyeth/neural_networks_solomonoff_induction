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

from neural_networks_solomonoff_induction.data import data_generator as dg_lib
from neural_networks_solomonoff_induction.data import utm_data_generator as utm_dg_lib
from neural_networks_solomonoff_induction.data import utms as utms_lib
from neural_networks_solomonoff_induction.models import transformer

alphabet_size = 2
probabilistic = True
length = 256

rng = np.random.default_rng(seed=1)

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
    batch_size=10000,
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
# for row in binary_batch:
#     print(row[:30])
#print(log_dict)

with open('binary_batch_log_dict.pkl', 'wb') as f:
    pickle.dump((binary_batch,log_dict),f)

if False:
    config = transformer.TransformerConfig(vocab_size=alphabet_size)
    model = hk.transform(
        functools.partial(transformer.transformer_decoder, config=config)
    )
    fname = "params.npz"
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
    print(true_conditionals)
    # still need to handle masking, see _make_loss_fn