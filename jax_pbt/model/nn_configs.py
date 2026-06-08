from abc import abstractmethod
from typing import Sequence

import flax.linen as nn
import flax.struct as struct
from flax.struct import PyTreeNode

from .nn_blocks import Activation, CNN, MLP


class NNConfig(PyTreeNode):
    @abstractmethod
    def create(self) -> nn.Module:
        ...

class CNNConfig(NNConfig):
    feature_lst: Sequence[int] = struct.field(default_factory=lambda: [64, 64, 64])
    kernel_lst: Sequence[Sequence[int]] = struct.field(default_factory=lambda: [(3, 3), (3, 3), (3, 3)])
    stride_lst: Sequence[int] = struct.field(default_factory=lambda: [1, 1, 1])
    activation_type: Activation.Type = 'relu'
    post_mlp_hidden_size: int = 64
    post_mlp_activation_type: Activation.Type = 'relu'
    
    def create(self) -> nn.Module:
        return nn.Sequential([
            CNN(
                feature_lst=self.feature_lst,
                kernel_lst=self.kernel_lst,
                stride_lst=self.stride_lst,
                activation_type=self.activation_type
            ),
            lambda x: x.reshape(*x.shape[:-3], -1),
            MLP(
                hidden_size_lst=[self.post_mlp_hidden_size],
                activation_type=self.post_mlp_activation_type
            )
        ])

class MLPConfig(NNConfig):
    hidden_layers: int = 2
    hidden_size: int = 64
    activation_type: Activation.Type = 'relu'

    def create(self) -> nn.Module:
        return MLP(
            hidden_size_lst=[self.hidden_size for _ in range(self.hidden_layers)],
            activation_type=self.activation_type
        )
