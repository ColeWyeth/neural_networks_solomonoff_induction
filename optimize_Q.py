import os
import pprint
from collections import defaultdict

import numpy as np
from typing import Tuple

from neural_networks_solomonoff_induction.data import utm_data_generator as utm_dg_lib
from neural_networks_solomonoff_induction.data import chronological_utm_data_generator as chronological_utm_dg_lib
from neural_networks_solomonoff_induction.data import utms as utms_lib

# Arbitrary constants from https://github.com/google-deepmind/neural_networks_solomonoff_induction/issues/5
# Increased repetition threshold since I am focused on binary alphabet
#REP_PERCENT_THRESH = 0.70 # paper version
#REP_PERCENT_THRESH = 0.95
REP_PERCENT_THRESH = 0.80
MAX_SHORT_PROGRAM_LEN = 100
INTERACTION_THRESH = 0.40

def repeating_count(output, delay):
  count = 0 # number of equal elements
  for i in range(delay + 1, len(output)):
    if output[i] == output[i-delay]: count += 1
  return count

def interesting(program_output: Tuple[str, str]):
    program, output = program_output
    max_count = 0
    for delay in range(1, len(output)//2):
       count = repeating_count(output, delay)
       max_count = max(max_count, count)
    if max_count > REP_PERCENT_THRESH * len(output):
       #print("too many repetitions")
       return False
    elif len(program) > MAX_SHORT_PROGRAM_LEN:
       #print("too long program")
       return False
    else:
       return True

def get_interacting_programs(log_dict):
   prog_results = []
   for res in log_dict['results']:
      output_len = len(res['ae'][0]) # this is the min of the outputs
      if res['actions_read'] > INTERACTION_THRESH*output_len and res['percepts_read'] > INTERACTION_THRESH*output_len:
         prog_results.append(res['agent_result'])
         prog_results.append(res['env_result'])
   return prog_results

if __name__ == "__main__":
   ctx_name = 'ctx_filtered_chron.pyd'
   mode = "CHRONOLOGICAL" # "SEQUENTIAL"
   examples = 20000 # divisible by batch size
   seq_length = 3000
   memory_size = 200
   maximum_steps = 10*seq_length
   maximum_program_length = 3000
   alphabet_size = 16

   batch_size = 100
   record_name = os.path.join('data', ctx_name)
   k = 2
   if mode == "SEQUENTIAL":
      tokens = '<>+-[]{.,'
   else:
      tokens = '<>+-[]{.,?'
   token_list = list(tokens)

   rng = np.random.default_rng(seed=1)
   program_sampler = utms_lib.FastSampler(rng=rng)

   def get_utm(is_chronological):
      return utms_lib.BrainPhoqueUTM(
         program_sampler,
         alphabet_size=alphabet_size,
         shorten_program=True,
         use_input_instruction=True,
         use_chronological_instruction=is_chronological,
      )

   if mode == "SEQUENTIAL":
      utm = get_utm(False)
      data_generator = utm_dg_lib.UTMDataGenerator(
         batch_size=batch_size,#1000,
         seq_length=seq_length,
         rng=rng,
         utm=utm,
         memory_size=memory_size,
         maximum_steps=maximum_steps,
         tokenizer=utm_dg_lib.Tokenizer.SEQ_POSITION,
         maximum_program_length=maximum_program_length,
      )
   else:
      utm_pair = utms_lib.ChronologicalUTMPair(
         agent_utm=get_utm(True),
         env_utm=get_utm(True),
      )
      data_generator = chronological_utm_dg_lib.ChronologicalUTMDataGenerator(
         batch_size=batch_size,
         seq_length=seq_length,
         rng=rng,
         utm_pair=utm_pair,
         memory_size=memory_size,
         maximum_steps=maximum_steps,
         tokenizer=utm_dg_lib.Tokenizer.SEQ_POSITION,
         maximum_program_length=maximum_program_length,
      )

   all_good_programs = []

   for i in range(examples//batch_size):
      seqs, log_dict = data_generator.sample()
      if mode == "SEQUENTIAL":
         results = log_dict['results']
      else:
         results = get_interacting_programs(log_dict)
      program_output_list = [(res['short_program'], res['output']) for res in results]
      # print(f"All programs and output: {len(program_output_list)}")
      # for po in program_output_list:
      #    print(f"{po[0]}: {repr(po[1])}")
      interesting_program_output_list = list(filter(interesting, program_output_list))
      interesting_programs = [po[0] for po in interesting_program_output_list]
      print(f"Batch {i} Interesting programs and output: {len(interesting_programs)}")
      for po in interesting_program_output_list:
         print(f"{po[0]}: {repr(po[1])}")
      all_good_programs.extend(interesting_programs)

   print(f"TOTAL Interesting programs and output: {len(all_good_programs)}")
   sym_to_ind = dict()
   for i, sym in enumerate(token_list):
      sym_to_ind[sym] = i

   def get_empty_counts():
      return len(token_list)*[0]
   counts = defaultdict(get_empty_counts)

   for p in all_good_programs:
      for i in range(k, len(p)):
         context = p[i-k:i]
         counts[context][sym_to_ind[p[i]]] += 1
   print(token_list)
   record = {
      'tokens': tokens,
      'counts_dict': dict(counts),
   }
   print(f"Saving record to {record_name}:")
   print(record)


   with open(record_name, 'w+') as f:
      f.write(pprint.pformat(record))

# The below is unnecessary, just count the occurences of each event.
# Passing this to the MC sampler it fits a Markov model (alpha = 0.5 corresponds to KT).
# def fit_markov(programs, symbols, k, step = 0.01):
#     sym_to_ind = dict()
#     for i, sym in enumerate(symbols):
#        sym_to_ind[sym] = i
#     # last index is for the next symbol
#     cond_probs = (1/len(symbols)) * np.ones((k+1)*[len(symbols)])

#     # Since the programs never change, we can once and for all
#     # count the number of occurences of each symbol in each context.
#     counts = np.zeros((k+1)*len(symbols))
#     for p in programs:
#         for i in range(k, len(p)):
#             sym_and_context = p[i-k:i+1] # include the final symbol
#             event_index = [sym_to_ind(s) for s in sym_and_context]
#             counts[tuple(event_index)] += 1

#     def proj_prob_simplex(y):
#         """See https://arxiv.org/abs/1309.1541 """
#         u = sorted(y, reverse=True)
#         rho = 0
#         j = 0
#         partial_sum = [0]
#         while rho < len(symbols):
#             j = j+1
#             partial_sum += u[j]
#             if u[j] + (1/j)*(1 - partial_sum) <= 0:
#                 break
#             else:
#                 rho = j
#         lmbda = (1/rho)(1 - sum(u[:rho+1]))
#         return [max(yi+lmbda,0) for yi in y]

#     while True:
#         # d/dx -ln x = -1/x
#         grad = np.multiply(-counts, 1/cond_probs)
#         if

#         # project the gradient to plane of simplex
#         proj_grad = grad - np.sum(grad, -1, keepdims=True)

#         # neg gradient step
#         cond_probs = cond_probs - (step * proj_grad)

#         # project each markov kernel to simplex
#         cond_probs = np.apply_along_axis(proj_prob_simplex, cond_probs, -1)

#         if (np.linalg.norm(grad, axis=-1) < 0.01).all():
#             break


#     def get_strings(symbols, length):
#         if length == 0:
#             return ""
#         strings = []
#         suffixes = get_strings(symbols, length-1)
#         for s in symbols:
#             strings.extend([s + string for string in suffixes])
#         return strings

#     contexts = get_strings(symbols, k-1)

#     for context in contexts:

#     return
