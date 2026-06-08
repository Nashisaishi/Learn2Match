import jax
import jax.numpy as jnp
import flax.linen as nn


class MultiLayerGRU(nn.Module):
    rnn_layers: int
    hidden_size: int
    use_layer_norm: bool
    """
    Applies the GRU unit step-by-step.
    
    Architecture:
        This module stacks multiple GRUCell layers sequentially:
            x_0 -> GRU_1 -> x_1 -> GRU_2 -> ... -> GRU_L -> x_L
        The carry is a concatenation of all layer states:
            carry = [h_1, h_2, ..., h_L]
        Only the top-layer output x_L is returned as the representation.

    [Input]
        - carry: Recurrent state before processing the step, with shape [*batch_shape, rnn_layers * hidden_size].
        - x: Input to the GRU, with shape [*batch_shape, input_dim].

    [Output]
        - carry: Updated recurrent state after processing the step, with shape [*batch_shape, rnn_layers * hidden_size].
        - x: Output (representation layer), with shape [*batch_shape, hidden_size].
    """

    def setup(self):
        self.layer_norms = [
            nn.LayerNorm() if self.use_layer_norm else lambda x: x
            for i in range(self.rnn_layers)
        ]
        self.gru_layers = [nn.GRUCell(features=self.hidden_size) for i in range(self.rnn_layers)]
    
    def __call__(self, carry: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        new_carry = []
        for i, (gru_layer, layer_norm) in enumerate(zip(self.gru_layers, self.layer_norms)):
            x = layer_norm(x)
            new_carry_layer, x = gru_layer(carry[..., i*self.hidden_size:(i+1)*self.hidden_size], x)
            new_carry.append(new_carry_layer)
        new_carry = jnp.concatenate(new_carry, -1)
        return new_carry, x

class RNN(nn.Module):
    rnn_layers: int = 1
    hidden_size: int = 64
    use_layer_norm: bool = False

    def setup(self):
        self.scan_gru = nn.scan(
            MultiLayerGRU, variable_broadcast="params",
            split_rngs={"params": False}, in_axes=0, out_axes=0
        )(
            rnn_layers=self.rnn_layers,
            hidden_size=self.hidden_size,
            use_layer_norm=self.use_layer_norm
        )
    
    def __call__(self, carry: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Applies the recurrent model over the input sequence.

        [Input]
            - carry: Recurrent state before processing the sequence, with shape [*batch_shape, rnn_layers * hidden_size].
            - x: Input sequence to the RNN, with shape [L, *batch_shape, input_dim].

        [Output]
            - carry: Updated recurrent state after processing the sequence, with shape [*batch_shape, rnn_layers * hidden_size].
            - x: Output sequence (representation layer), with shape [L, *batch_shape, hidden_size].
        """
        carry, x = self.scan_gru(carry, x)
        return carry, x
    