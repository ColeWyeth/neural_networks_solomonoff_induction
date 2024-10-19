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

"""Transformer model."""

import dataclasses

import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import functools

@dataclasses.dataclass(kw_only=True)
class TransformerConfig:
  """Hyperparameters used in the Transformer architectures."""

  # Vocabulary size.
  vocab_size: int
  # The dimension of the first embedding.
  embedding_dim: int = 256 # 32
  # The number of multi-head attention layers.
  num_layers: int = 6 # 6 my largest model has two extra layers 6+2=8
  # The number of heads per layer.
  num_heads: int = 4 #8
  # The parameter initialization scale for the embeddings.
  emb_init_scale: float = 0.02
  # How much larger the hidden layer of the feedforward network should be
  # compared to the `embedding_dim`.
  widening_factor: int = 4


class MultiHeadDotProductAttention(hk.Module):
  """Multi-head dot-product attention (Vaswani et al., 2017)."""

  def __init__(
      self,
      num_heads: int,
      num_hiddens_per_head: int,
      name: str | None = None,
  ) -> None:
    """Initializes the attention module.

    Args:
      num_heads: Number of heads to use.
      num_hiddens_per_head: Number of hidden neurons per head.
      name: Name of the module.
    """
    super().__init__(name=name)
    self._num_heads = num_heads
    self._num_hiddens_per_head = num_hiddens_per_head
    self.num_hiddens = self._num_hiddens_per_head * self._num_heads
    self.lin_q = hk.Linear(self.num_hiddens, with_bias=False)
    self.lin_k = hk.Linear(self.num_hiddens, with_bias=False)
    self.lin_v = hk.Linear(self.num_hiddens, with_bias=False)
    self.lin_out = hk.Linear(self.num_hiddens, with_bias=False) # This assumes embedding_dim = num_hiddens
  def __call__(
      self,
      inputs_q: jax.Array,
      inputs_kv: jax.Array,
      mask: jax.Array | None = None,
  ) -> jax.Array:
    """Returns the output of the multi-head attention."""
    batch_size, sequence_length, embedding_size = inputs_q.shape

    q = self.lin_q(inputs_q)
    k = self.lin_k(inputs_kv)
    v = self.lin_v(inputs_kv)
    # print(f"q first: {q[0,:,0]}")
    # print(f"k first: {k[0,:,0]}")
    # print(f"v first: {v[0,:,0]}")
    # The second (sequence) dimension is undefined since it can differ between
    # queries and keys/values when decoding. Also checking that the inputs have
    # the same batch size as the reshape below does not guarantee a failure if
    # they are different.
    new_shape = (batch_size, -1, self._num_heads, self._num_hiddens_per_head)
    q = jnp.reshape(q, new_shape)
    k = jnp.reshape(k, new_shape)
    v = jnp.reshape(v, new_shape)

    # Let b=batch_size, t=seq_len, h=num_heads, and d=num_hiddens_per_head.
    attention = jnp.einsum('bthd,bThd->bhtT', q, k)
    attention *= 1.0 / jnp.sqrt(self._num_hiddens_per_head)
    # print(f"attention: {attention}")

    if mask is not None:
      attention = jnp.where(mask, attention, jnp.finfo(jnp.float32).min)

    normalized_attention = jnn.softmax(attention)
    # print(f"normalized attention: {normalized_attention}")

    output = jnp.einsum('bhtT,bThd->bthd', normalized_attention, v)
    output = jnp.reshape(output, (batch_size, sequence_length, self.num_hiddens))
    #print(f"output: {output}")
    return self.lin_out(output)

  def inference(
      self,
      inputs_q: jax.Array,
      inputs_kv: jax.Array,
      mem_k: jax.Array,
      mem_v: jax.Array,
  ) -> jax.Array:

    q = self.lin_q(inputs_q)
    new_k = self.lin_k(inputs_kv)
    new_v = self.lin_v(inputs_kv)
    k = jnp.concatenate(
      [mem_k, jnp.expand_dims(new_k,0)],
      0,
    )
    v = jnp.concatenate(
      [mem_v, jnp.expand_dims(new_v,0)],
      0,
    )
    # print(f"q first: {q[0]}")
    # print(f"k first: {k[:,0]}")
    # print(f"v first: {v[:,0]}")

    q = jnp.reshape(q, (self._num_heads, self._num_hiddens_per_head))
    new_shape = (-1, self._num_heads, self._num_hiddens_per_head)
    k = jnp.reshape(k, new_shape)
    v = jnp.reshape(v, new_shape)

    # Let b=batch_size, t=seq_len, h=num_heads, and d=num_hiddens_per_head.
    attention = jnp.einsum('hd,Thd->hT', q, k)
    attention *= 1.0 / jnp.sqrt(self._num_hiddens_per_head)
    # print(f"attention: {attention}")

    # Causal masking is unnecessary because we only need activations for the latest token

    normalized_attention = jnn.softmax(attention)
    # print(f"normalized attention: {normalized_attention}")

    output = jnp.einsum('hT,Thd->hd', normalized_attention, v)
    output = jnp.reshape(output, (self.num_hiddens,))
    #print(f"output: {output}")
    return self.lin_out(output), new_k, new_v

# class Embedding(hk.Module):
#   def __init__(
#       self,
#       config: TransformerConfig,
#   ):
#     self.config = config
#     self.embs_init = hk.initializers.TruncatedNormal(stddev=config.emb_init_scale)
#     self.embeddings_layer = hk.Embed(
#         vocab_size=config.vocab_size,
#         embed_dim=config.embedding_dim,
#         lookup_style=hk.EmbedLookupStyle.ARRAY_INDEX,
#         w_init=self.embs_init,
#     )
def sinusoid_position_encoding(
    pos_seq: int | np.ndarray,
    hidden_size: int,
    max_timescale: float = 1e4,
) -> np.ndarray:
  """Creates sinusoidal encodings from the original transformer paper.

  The returned values are, for all i < D/2:
    array[pos, i] = sin(pos / (max_timescale^(2*i / D)))
    array[pos, D/2 + i] = cos(pos / (max_timescale^(2*i / D)))

  Args:
    sequence_length: Sequence length.
    hidden_size: Dimension of the positional encoding vectors, D. Should be
      even.
    max_timescale: Maximum timescale for the frequency.

  Returns:
    An array of shape [L, D] if `add_negative` or `keep_positive_side` is
    `False`, else [2 * L, D].
  """
  freqs = np.arange(0, hidden_size + 1, 2)
  inv_freq = max_timescale ** (-freqs / hidden_size)

  if type(pos_seq) == int:
    pos_seq = np.arange(start=0, stop=pos_seq)

  sinusoid_inp = np.einsum('i,j->ij', pos_seq, inv_freq)
  embeddings = np.concatenate(
      [np.sin(sinusoid_inp), np.cos(sinusoid_inp)], axis=-1
  )
  return embeddings[:, :hidden_size]

def embed_sequences(
    sequences: jax.Array,
    config: TransformerConfig,
    pos_seq: np.ndarray | None = None,
) -> jax.Array:
  """Returns embeddings for sequences of tokens."""
  embs_init = hk.initializers.TruncatedNormal(stddev=config.emb_init_scale)
  embeddings_layer = hk.Embed(
      vocab_size=config.vocab_size,
      embed_dim=config.embedding_dim,
      lookup_style=hk.EmbedLookupStyle.ARRAY_INDEX,
      w_init=embs_init,
  )

  embeddings = embeddings_layer(sequences)
  embeddings *= jnp.sqrt(config.embedding_dim)

  _, sequence_length, embedding_size = embeddings.shape
  # If the indices weren't passed, assume this is the full sequence.
  # The positional encoding will reconstruct indices.
  if pos_seq is None:
    pos_seq = sequence_length
  pos_encodings = sinusoid_position_encoding(
      pos_seq=pos_seq,
      hidden_size=embedding_size,
  )
  return embeddings + pos_encodings


def layer_norm(x: jax.Array) -> jax.Array:
  """Helper function for layer norm."""
  return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)


def shift_right(sequences: jax.Array) -> jax.Array:
  """Right-shift the one-hot encoded input by padding on the temporal axis."""
  bos_array = jnp.zeros((sequences.shape[0], 1), dtype=jnp.uint8)
  padded_sequences = jnp.concatenate([bos_array, sequences], axis=1)
  return padded_sequences[:, :-1]


def transformer_decoder(
    targets: jax.Array,
    config: TransformerConfig,
) -> jax.Array:
  """Returns the transformer decoder output, shape [B, T, V].

  Args:
    targets: The integer target values, shape [B, T].
    config: The config to use for the transformer.
  """
  # Right shift the targets to get the inputs (the first token is now a 0).
  inputs = shift_right(targets)

  # Embeds the inputs and adds positional encodings.
  embeddings = embed_sequences(inputs, config)
  # print(f"embeddings: {embeddings}")

  batch_size, sequence_length = embeddings.shape[:2]

  # The causal mask is shared across heads.
  causal_mask = np.tril(
      np.ones((batch_size, 1, sequence_length, sequence_length))
  )

  h = embeddings
  for _ in range(config.num_layers):
    self_attention = MultiHeadDotProductAttention(
        num_heads=config.num_heads,
        num_hiddens_per_head=config.embedding_dim // config.num_heads,
    )(inputs_q=h, inputs_kv=h, mask=causal_mask)
    # if i == 0:
    #   print(f"self attention: {self_attention}")
    attention = layer_norm(h + self_attention)

    # Position-wise feedforward network.
    h = hk.Linear(config.embedding_dim * config.widening_factor)(attention)
    h = jnn.gelu(h)
    h = hk.Linear(config.embedding_dim)(h)
    h = layer_norm(h + attention)

  logits = hk.Linear(config.vocab_size)(h)
  return jnn.log_softmax(logits, axis=-1)

# def markov_kernel(
#     input_q,
#     input_kv,
#     mem_k,
#     mem_v,
#     config,
# ):
#   return MultiHeadDotProductAttention(
#     num_heads=config.num_heads,
#     num_hiddens_per_head=config.embedding_dim // config.num_heads,
#   ).inference(input_q, input_kv, mem_k, mem_v)

def markov_kernel(
    h, # new token embedding
    mem_k,
    mem_v,
    config,
):
  """Input is not right shifted so should always pass in embedding of 0 first"""
  # embedding = embed_sequences([input], config)

  # print(f"embedding: {embedding}")

  #h = embedding[0, -1, :] # we're only interested in latest token embedding
  new_k, new_v = [], []
  for i in range(config.num_layers):
    gpu = jax.devices('gpu')[0]
    self_attention, new_k_i, new_v_i = MultiHeadDotProductAttention(
      num_heads=config.num_heads,
      num_hiddens_per_head=config.embedding_dim // config.num_heads,
    ).inference(h, h, jax.device_put(mem_k[i], device=gpu), jax.device_put(mem_v[i], device=gpu)) # TODO: put these on the device as needed here.
    # if i == 0:
    #   print(f"self_attention: {self_attention}")
    attention = layer_norm(h + self_attention)
    new_k.append(new_k_i)
    new_v.append(new_v_i)

    # Position-wise feedforward network.
    h = hk.Linear(config.embedding_dim * config.widening_factor)(attention)
    h = jnn.gelu(h)
    h = hk.Linear(config.embedding_dim)(h)
    h = layer_norm(h + attention)

  logits = hk.Linear(config.vocab_size)(h)
  return jnn.log_softmax(logits, axis=-1), jnp.array(new_k), jnp.array(new_v)

class Memoized_Transformer:
  def __init__(self, params, config):
    self.params = params
    self.config = config
    self.kernel = hk.transform(
      functools.partial(markov_kernel, config=config),
    )
    self.embed = hk.transform(
      functools.partial(embed_sequences, config=config)
    )
    # This assumes embedding dim is num hiddens.
    self.mem_k = jnp.zeros((config.num_layers, 0, config.embedding_dim))
    self.mem_v = jnp.zeros((config.num_layers, 0, config.embedding_dim))
    self.outputs = []
    self.seq = []
    self.update(0) # all sequences start with a 0 during training
  def update(self, symbol):
    self.seq.append(symbol)
    embedding = self.embed.apply(
      params = self.params,
      sequences = jnp.array([[symbol]]),
      pos_seq = np.array([len(self.seq)]),
      rng = None,
    )
    h = embedding[0, 0, :]
    output, new_k, new_v = self.kernel.apply(
      params = self.params,
      h = h,
      mem_k = self.mem_k,
      mem_v = self.mem_v,
      rng = None,
    )
    self.outputs.append(output)
    # TODO: first remove new_k and new_v from the device.
    cpu = jax.devices('cpu')[0]
    self.mem_k = jnp.concatenate([self.mem_k, jnp.expand_dims(jax.device_put(new_k, cpu), 1)], 1)
    self.mem_v = jnp.concatenate([self.mem_v, jnp.expand_dims(jax.device_put(new_v, cpu), 1)], 1)
  def erase(self, n):
    self.seq = self.seq[:-n]
    self.mem_k = self.mem_k[:, :-n, :]
    self.mem_v = self.mem_v[:, :-n, :]
    self.outputs = self.outputs[:-n]
  def predict(self):
    return self.outputs[-1]
