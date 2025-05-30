# %%
import numpy as np

import sys
nnsi_path = "/root/neural_networks_solomonoff_induction/.."
if f"{nnsi_path}" not in sys.path:
    sys.path.insert(0,f"{nnsi_path}")

from neural_networks_solomonoff_induction.data import data_generator as dg_lib
from neural_networks_solomonoff_induction.data import utm_data_generator as utm_dg_lib
from neural_networks_solomonoff_induction.data import utms as utms_lib
from neural_networks_solomonoff_induction.models import transformer
from neural_networks_solomonoff_induction.models.ctw import CTWPredictor


depth = 15
alphabet_size = 2
probabilistic = True
length = 256

generate_fresh_batch = True
num_examples = 1000
# visualize = True

# %%
# Sample a training batch of programs
rng = np.random.default_rng(seed=9) # NOT 8

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
    batch_size=num_examples,
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

# %%
predictor = CTWPredictor(depth)
params = {}
predictions, _, new_params, _, loss_dict = predictor.update(params, batch)
# %%
import pickle
with open('ctw_params.pkl','wb') as f:
    pickle.dump(new_params, f)
# %%
