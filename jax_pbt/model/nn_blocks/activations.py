from typing import Callable, Literal

import flax.linen as nn
import jax


class Activation:
    Type = Literal[
        'relu', 'tanh', 'sigmoid', 'swish', 'gelu'
    ]

    table = {
        'relu': nn.relu,
        'tanh': nn.tanh,
        'sigmoid': nn.sigmoid,
        'swish': nn.swish,
        'gelu': nn.gelu,
    }

    @staticmethod
    def get(name: Type) -> Callable[[jax.Array], jax.Array]:
        return Activation.table[name]
