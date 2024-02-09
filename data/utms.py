# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple universal turing machines, used for generating data.

This file contains the interface for the UTMs (an abstract class), and the code
for the BrainPhoque machine (an implementation of the interface), operating on
bits. The goal is to generate data from programs sampled from the Solomonoff
prior.
"""

import abc
from typing import Mapping, Sequence, Union

import numpy as np


class UniversalTuringMachine(abc.ABC):
  """An abstract class for our universal Turing machines."""

  @property
  @abc.abstractmethod
  def program_tokens(self) -> Sequence[str]:
    """Returns the tokens that can be used in a program."""

  @property
  @abc.abstractmethod
  def alphabet_size(self) -> int:
    """Returns the size of the alphabet of the data and outputs."""

  @abc.abstractmethod
  def run_program(
      self,
      program: str,
      memory_size: int,
      maximum_steps: int,
      max_output_length: int,
  ) -> Mapping[str, Union[int, str]]:
    """Returns an output string from a program run on the machine.

    Args:
      program: The program to run on the UTM. It is a string of concatenated
        tokens from `self.program_tokens`.
      memory_size: The size of the memory to use.
      maximum_steps: Maximum number of steps after which the function will stop.
      max_output_length: Maximum length of the output sequence.

    Raises:
      IncorrectProgramError: If the input program is incorrect (see examples
        in exception definition below).
    """

  def sample_program(
      self,
      length: int,
      rng: int | np.random.Generator = 1,
  ) -> str:
    """Returns a program of a given length, with uniformly sampled tokens.

    Args:
      length: The length of the program which will be sampled.
      rng: The numpy random generator to use, or the integer seed. Allows to
        reuse the same generator for multiple usage.
    """
    if isinstance(rng, int):
      rng = np.random.default_rng(rng)
    return ''.join(rng.choice(self.program_tokens, length))


class IncorrectProgramError(Exception):
  """A program cannot be run on the UTM.

  Examples:
    * Some tokens are not part of `utm.program_tokens`.
    * Bad indentation (in Python for instance).
  """


class BrainPhoqueUTM(UniversalTuringMachine):
  """BrainPhoque (Brainf*** from its real name) universal turing machine.

  Reference: https://en.wikipedia.org/wiki/Brainfuck
  This machine only uses 7 program characters, described in the link above. The
  machine uses a tape memory, filled with integers within the range
  {first_data_int, ... last_data_int}, which can be incremented or
  decremented with the '+' and '-' instructions, with wrap-around when reaching
  the boundaries of the interval.
  Moving the reading head on the data tape is done with the '>' and
  '<' instructions. Loops are created using the '[' and ']' instructions.
  By contrast to the original BrainFuck design, '[' does not jump at all,
  while ']' jumps back to '[' if the integer under the reading head is *not* 0.
  This is an important difference and Turing completeness has not been proven
  for this yet (seems like it should hold though).
  Finally, outputs are returned with the instruction '.'.
   Note that we omit the ',' which is used to read from the input.
  See the reference above for more details.

  Note about the probabilities of the output strings generated by the
  programs (bear with me, this is a little subtle):
  One may think of using the prior weight of the generated program to
  obtain a lower bound on the output sequence generated by this program.
  However, this is pretty loose compared to the cumulative weight of all
  the programs that output the same sequence (and possibly more) --- which
  is the actual quantity we care about.
  Instead, every program on our machine is assumed to be infinite(!). The
  BrainPhoque machine ensures that every program is valid.
  The prior of a given program is A^{-n} with n->oo for a program alphabet
  of size A.
  The program are not prefix-free, but the sum of the prior weights is still
  (exactly) 1.
  Let Q be the set of all programs, let w_q be the prior weight of a
  program q.
  Suppose we run program p for T steps and we obtain output s (≤ T).
  Let N be the number of different instructions of the program p that have
  been evaluated during the T steps (N ≤ T).
  Then the Solomonoff mixture probability for s is
  M(s) = sum_{q in Q:U(q)=s*} w_q  ≥  A^{-N}

  Note that 'utm.run_program' returns a shortened program of length at most N.

  The current implementation of the BrainPhoque UTM ensures that the position
  of the last evaluated instruction is no more than T (which may not be the
  case when opening brackets can jump to an arbitrary location).
  Hence, if _MAX_PROGRAM_LENGTH ≥ T, we know that we are not 'missing'
  obvious programs that would generate the same string.
  By contrast, a BrainFuck (not BrainPhoque) program that has fewer than
  _MAX_PROGRAM_LENGTH has an undefined behaviour if an opening bracket
  is unmatched; in BrainPhoque, since opening brackets are always skipped,
  the behaviour is well-defined.
  """

  def __init__(self, alphabet_size: int = 17):
    """Constructor.

    Args:
      alphabet_size: The size of the data and output alphabets.
    """
    self._alphabet_size = alphabet_size

  @property
  def alphabet_size(self) -> int:
    """Returns the size of the data and output alphabets."""
    return self._alphabet_size

  @property
  def program_tokens(self) -> Sequence[str]:
    """Returns the tokens that can be used to write a BrainPhoque program."""
    return ['+', '-', '>', '<', '[', ']', '.']

  def _get_matching_brackets_from_program(self, program: str) -> dict[int, int]:
    """Returns matching brackets from a BrainPhoque program.

    Unmatched bracket point to their own position.

    This function is part of the object and not outside for readability.

    Example:
      +[[+<+]+] will return the dict {1: 8, 8: 1, 2: 7, 7: 2}, as the first open
      bracket (at position 1) matches with the last bracket (position eight),
      and same for the brackets at positions 2 and 7.

    Args:
      program: The BrainPhoque program to analyze.
    """
    matching_brackets = {}
    stack = []
    for index, command in enumerate(program):
      match command:
        case '[':
          stack.append(index)
        case ']':
          if stack:
            last_index = stack.pop()
            matching_brackets[last_index] = index
            matching_brackets[index] = last_index
          else:
            # Unmatched closing bracket, just skip it.
            matching_brackets[index] = index
    while stack:
      # Unmatched opening brackets, just skip them.
      index = stack.pop()
      matching_brackets[index] = index
    return matching_brackets

  def run_program(
      self,
      program: str,
      memory_size: int,
      maximum_steps: int,
      max_output_length: int,
  ) -> Mapping[str, Union[int, str]]:
    """Returns the output of a program on the BrainPhoque UTM.

    Args:
      program: See base class.
      memory_size: The size of the (tape) memory to use. If the boundaries are
        reached by the memory pointer, we cycle back to the other end.
      maximum_steps: See base class.
      max_output_length: maximum length of the output sequence.

    Raises:
      IncorrectProgramError: If some tokens of the program are not in
        `self.program_tokens`.
    """
    # The index of the pointer moving on the program string.
    program_index = 0
    # The memory tape, initialised with all zeros. Can only contain integers
    # between 0 and alphabet size - 1.
    memory = [0] * memory_size
    # The index of the pointer moving on the memory tape.
    memory_index = 0
    # We compute the matching brackets in the program once at the beginning.
    matching_brackets = self._get_matching_brackets_from_program(program)
    # The output is a string.
    output = ''
    # We count the number of steps to stop if it's too high.
    num_steps = 0
    # Keep track of which instructions have been used
    used_instructions = [0] * len(program)

    def make_result(status) -> Mapping[str, Union[int, str]]:
      # Build an equivalent program that contains only the instructions
      # that have been evaluated at least once.
      short_program = ''
      for instruction, instruction_use_count in zip(program, used_instructions):
        if instruction_use_count != 0:
          short_program += instruction

      return {
          'status': status,
          'alphabet_size': self._alphabet_size,
          'num_steps': num_steps,
          'memory_index': memory_index,
          'output': output,
          'output_length': len(output),
          'short_program': short_program,
          'short_program_length': len(short_program),
      }

    while program_index < len(program):
      command = program[program_index]
      used_instructions[program_index] += 1
      mem = memory[memory_index]

      match command:
        case '+':
          # Increment data cell value with wrap-around
          # Multiple behaviours are possible:
          # a. Wrapping around from last_data_int to first_data_int
          #    The 'problem' (for mhutter) is that this can generate
          #    counters based on the size of the alphabet. However it's more
          #    consistent with the 'invert' operation of BoolPhoque.
          # b. The data tape can hold arbitrary integers but the output is
          #    modulo'ed to [first_data_int:last_data_int].
          #    This feels less natural to me (lorseau) in particular because the
          #    individual tape cells are unbounded in memory.
          # c. Clipping: values are clipped to the range
          #    [first_data_int:last_data_int].
          #    Pb: programs on average are significantly less interesting
          #    because negative numbers are disallowed, making (more than)
          #    half of the programs output constant values.
          memory[memory_index] = (mem + 1) % self._alphabet_size
        case '-':
          # Decrement data cell value with wrap-around.
          memory[memory_index] = (mem - 1) % self._alphabet_size
        case '.':
          # Output command: we append to the output string.
          # We use a string for convenience, but it should be view as a
          # bytearray instead.
          output += chr(mem)
        case '<':
          # Move left on the tape.
          memory_index = (memory_index - 1) % memory_size
        case '>':
          # Move right on the tape.
          memory_index = (memory_index + 1) % memory_size
        case '[':
          # Specific to BrainPhoque (by contrast to BrainFuck): open brackets
          # do not jump. See utm_data_generator.sample_params for the reason.
          pass
        case ']':
          # Go to the matching open bracket if current value is not zero.
          if mem != 0:
            program_index = matching_brackets[program_index]
            used_instructions[program_index] += 1
        case _:
          raise IncorrectProgramError(
              f'Character {command} is not recognized. All '
              'characters in the input program must be part of the set ('
              f'{",".join(self.program_tokens)}).',
          )
      program_index += 1
      num_steps += 1
      if program_index == len(program):
        return make_result('HALTED')
      if num_steps == maximum_steps:
        return make_result('TIMEOUT')
      if len(output) >= max_output_length:
        return make_result('OUTPUT_LIMIT')

    return make_result('INVALID')  # This should never be reached
