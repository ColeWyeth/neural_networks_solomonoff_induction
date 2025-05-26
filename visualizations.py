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
    batch_size=50,
    seq_length=256,
    rng=rng,
    utm=utm,
    memory_size=200, # matches optimize_Q
    maximum_steps=1024, # matches optimize_Q
    tokenizer=utm_dg_lib.Tokenizer.SEQ_POSITION,
    maximum_program_length=100,
)

batch, log_dict = data_generator.sample()
batch = np.array(batch)

print(batch.shape)
binary_batch = batch.argmax(axis=-1)
print(binary_batch.shape)
#print(f"{binary_batch=}")
for row in binary_batch:
    print(row[:30])
#print(log_dict)