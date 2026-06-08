from typing import Sequence

import jax
import flax.linen as nn

from .activations import Activation


class CNN(nn.Module):
    feature_lst: Sequence[int]
    kernel_lst: Sequence[Sequence[int]]
    stride_lst: Sequence[int]
    activation_type: Activation.Type = 'relu'
    # TODO brief and intruive despction of nn architecure and exptected input and output
    def setup(self):
        self.cnn_layers = [
            nn.Conv(
                features=num_features,
                kernel_size=kernel_shape,
                strides=stride
            )
            for (num_features, kernel_shape, stride) in zip(self.feature_lst, self.kernel_lst, self.stride_lst)
        ]
        self.activation_fn = Activation.get(self.activation_type)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        [Input]
            - x: [*batch_shape, W, H, C]
        
        [Output]
            - x: Final layer with shape[*batch_shape, W', H', feature_lst[-1]].
        """
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
            x = self.activation_fn(x)
        return x
