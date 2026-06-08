from abc import ABC, abstractmethod

from flax.struct import PyTreeNode
import jax


class BaseTrainer(ABC):
    @abstractmethod
    def init_trainer_state(self, rng: jax.Array) -> PyTreeNode:
        """
        Initialize the trainer state.
        """
        ...
    
    @abstractmethod
    def model_state_from_trainer_state(self, trainer_state: PyTreeNode) -> PyTreeNode:
        """
        Retreive the model state from the trainer state.
        """
        ...
