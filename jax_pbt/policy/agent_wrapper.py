from abc import ABC, abstractmethod
from typing import Sequence

from flax.struct import PyTreeNode
import jax

from ..env.spaces import Observation, Action


class BaseAgent(ABC):
    """Abstract base class defining the agent interface for interaction with the environment."""

    @abstractmethod
    def step(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        obs: Observation
    ) -> tuple[Action, dict]:
        """
        Perform a single step in the environment given the current model state and observation.

        [Input]
            - rng: Random number generator state for stochastic policies.
            - model_state: The agent's current model state.
            - obs: The observation received from the environment.

        [Output]
            action: The action selected by the agent.
            extra_data: Optional extra data, could be an empty dict.
        """
        ...


class RecurrentAgent(BaseAgent):
    """
    Abstract agent class for models with recurrent state dependencies (e.g., RNN-based agents).
    """

    @abstractmethod
    def init_agent_state(self, batch_shape: Sequence[int] = ()) -> PyTreeNode:
        """
        Initialize a batch of recurrent agent states (e.g., RNN hidden states).

        [Input]
            - batch_shape: Shape of the batch for initialization.

        [Output]
            - agent_state: A batch of initialized agent states.
        """
        ...

    @abstractmethod
    def reset_agent_state(self, agent_state: PyTreeNode, done: jax.Array) -> PyTreeNode:
        """
        Reset agent states at positions where episodes are done. The shape of agent_state and done should match.

        [Input]
            - agent_state: The current recurrent agent states.
            - done: Boolean array indicating which agent states should be reset.

        [Output]
            - reset_agent_state (PyTreeNode): Updated agent states with resets applied.
        """
        ...
    
    @abstractmethod
    def step(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        agent_state: PyTreeNode,
        obs: Observation
    ) -> tuple[PyTreeNode, Action, dict]:
        """
        Perform a single step in the environment with recurrent state updates.

        [Input]
            - rng: Random number generator state for stochastic policies.
            - model_state: The agent's current model state.
            - agent_state: The agent's recurrent state (e.g., RNN hidden states).
            - obs: The observation received from the environment.

        [Output]
            - next_agent_state: Updated agent state (e.g., next RNN hidden state).
            - action: The action selected by the agent.
            - extra_data: Optional extra data, could be an empty dict.
        """
        ...


class RecurrentPPOAgent(RecurrentAgent):
    """
    Abstract agent class for a Recurrent Proximal Policy Optimization (PPO) agent.

    This class extends `RecurrentAgent` by incorporating policy gradient learning.
    The agent maintains and updates recurrent hidden states while also computing 
    log probabilities and value estimates, which are essential for PPO updates.
    """

    @abstractmethod
    def step(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        agent_state: PyTreeNode,
        obs: Observation
    ) -> tuple[PyTreeNode, Action, dict[str, jax.Array]]:
        """
        Perform a single step in the environment with recurrent state updates and additional outputs for policy optimization.

        [Input]
            - rng: Random number generator state for stochastic policies.
            - model_state: The agent's current model state.
            - agent_state: The agent's recurrent state (e.g., RNN hidden states).
            - obs: The observation received from the environment.
        
        The batch shape of agent_state and obs should be the same.

        [Output]
            - next_agent_state: Updated agent state (e.g., next RNN hidden state).
            - action: The action selected by the agent.
            - extra_data: Dict with keys:
                - log_p: Log probability of the selected action.
                - val: Estimated value function output at the current observation.
        """
        ...
