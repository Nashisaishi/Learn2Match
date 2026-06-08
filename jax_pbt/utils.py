import argparse
from typing import Sequence

import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode


def parse_tuple(s):
    """Parse a string to a tuple of integers. Example input: '3,3'."""
    try:
        return tuple(map(int, s.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple must be in the format 'int,int'")

def pytree_repeat_stack(node: PyTreeNode, batch_shape: Sequence[int]):
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (*batch_shape, *(x.shape if hasattr(x, 'shape') else ()))),
        node
    )

def rng_batch_split(rng: jax.Array, batch_size: int | Sequence[int] = 1) -> tuple[jax.Array, jax.Array]:
    rng, rng_batch = jax.random.split(rng)
    rng_batch = jax.random.split(rng_batch, batch_size)
    return rng, rng_batch

def split_into_minibatch(rng: jax.Array, data: PyTreeNode, chunk_length: int, num_minibatches: int | None = 1, minibatch_num_chunks: int | None = None) -> PyTreeNode:
    """
    [Input]
        - data: [episode_length, batch_size, *data_attribute_shape]
    
    [Output]
        - minibatches: [num_minibatches, chunk_length, minibatch_num_chunks, *data_attribute_shape]
    
    [Notation]
        - L: episode_length
        - L': chunk_length
        - B: batch_size
        - B': num_minibatches
        - A: *attribute_shape
    """
    def reshape_data(data: jax.Array):
        # data: [L, B, A]
        episode_length = data.shape[0]
        chunks_per_episode = episode_length // chunk_length
        data = data[:chunks_per_episode*chunk_length].reshape(chunks_per_episode, chunk_length, *data.shape[1:]) # [B', L', B, A]
        data = jnp.swapaxes(data, 1, 2).reshape(-1, chunk_length, *data.shape[3:]) # [BB', L', A]
        p = jax.random.permutation(rng, data.shape[0])
        data = jnp.take(data, p, axis=0)
        n = data.shape[0] // minibatch_num_chunks if minibatch_num_chunks is not None else num_minibatches
        data = data[:data.shape[0]//n*n].reshape(n, -1, *data.shape[1:]) # [N, BB'/N, L, A]
        return jnp.swapaxes(data, 1, 2) # [N, L', BB'/N, A]
    return jax.tree_util.tree_map(reshape_data, data)

def global_norm(params: PyTreeNode) -> jax.Array:
    return jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(params)))
