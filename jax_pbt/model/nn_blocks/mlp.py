from typing import Sequence

import jax
import flax.linen as nn

from .activations import Activation


class MLP(nn.Module):
    hidden_size_lst: Sequence[int]
    activation_type: Activation.Type = 'relu'
    # TODO brief and intruive despction of nn architecure and exptected input and output

    def setup(self):
        self.fc_layers = [nn.Dense(hidden_size) for hidden_size in self.hidden_size_lst]
        self.activation_fn = Activation.get(self.activation_type)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        [Input]
            - x: [*batch_shape, input_dim]
        
        [Output]
            - x: Finaly layer with shape [*batch_shape, hidden_size_lst[-1]]
        """
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = self.activation_fn(x)
        return x