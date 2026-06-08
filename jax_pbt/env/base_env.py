from abc import ABC, abstractmethod
from typing import Any, Generic, Sequence, TypeVar

from flax.struct import PyTreeNode
import jax

from .spaces import Observation, Action, ObservationSpace, ActionSpace


class BaseEnvConst(PyTreeNode):
    # Stores the constant variables of the environment, such as maximum steps allowed per episode.
    max_steps: int

class BaseEnvState(PyTreeNode):
    # Represents the internal state of the environment that changes each step, reflecting episode dynamics from actions and random events.
    _step: int

EnvConstType = TypeVar('EnvConstType', bound='BaseEnvConst')
EnvStateType = TypeVar('EnvStateType', bound='BaseEnvState')

class BaseEnv(ABC, Generic[EnvConstType, EnvStateType]):
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    @property
    @abstractmethod
    def default_const(self) -> EnvConstType:
        # Returns the default constant configuration of the environment.
        ...
    
    @property
    @abstractmethod
    def env_observation_space(self) -> tuple[ObservationSpace, dict[str, tuple[int]]]:
        # The shape/space of observation attributes, and the shape/number of agents for each observation attribute
        ...
    
    @property
    @abstractmethod
    def env_action_space(self) -> tuple[ActionSpace, dict[str, tuple[int]]]:
        # The shape/space of action attributes, and the shape/number of agents for each action attribute
        ...

    @abstractmethod
    def env_reset(
        self,
        rng: jax.Array,
        const: EnvConstType
    ) -> tuple[EnvStateType, Observation]:
        """
        Resets the internal environment state and returns the initial observation.

        [Input]
            - rng: Random number generator array used for stochastic initialization.
            - const: Environment constants that configure the initial state.

        [Output]
            - state: Initial environment state after reset.
            - observation: Initial observation returned by the environment.
        """
        ...
    
    @abstractmethod
    def env_step(
        self,
        rng: jax.Array,
        const: EnvConstType,
        state: EnvStateType,
        action: Action
    ) -> tuple[EnvStateType, Observation, jax.Array, jax.Array, dict[Any, Any]]:
        """
        Represents the internal dynamics of taking a single step in the environment using the provided action.
        This is an internal API that handles the core environment logic.
        Note that automatic environment reset is expected to be implemented here.

        [Input]
            - rng: Random number generator array for random events.
            - const: Environment constants that configure the step behavior.
            - state: Current state of the environment.
            - action: Action to be taken by the agents.

        [Output]
            - state: The new environment state after applying the action.
            - observation: The observation produced by the environment at the next step.
            - reward: Reward values for the step.
            - done: Boolean flags indicating episode termination.
            - info: Additional information returned by the environment.

        Note for recommended implementation for maximum flexibility:
            The observation can be represented as a dictionary where similar attributes of the observation are grouped together.
            For example:
            Observation({
                "obs_group_1": [number of env-agents for obs_group_1, *observation_shape_1],
                "obs_group_2": [number of env-agents for obs_group_2, *observation_shape_2]
            })
            Here env-agents are internal agents in the environments, as opposed to the MARL trained agents (i.e, policies).
        """
        ...
        
    @property
    @abstractmethod
    def num_agents(self) -> int:
        # Returns the number of agents in the environment. Exposed as an API for RL algorithms
        ...

    @abstractmethod
    def get_observation_space(self) -> list[ObservationSpace]:
        # Returns a list of observation spaces for each agent as a public API for RL
        ...
    
    @abstractmethod
    def get_action_space(self) -> list[ActionSpace]:
        # Returns a list of action spaces for each agent as a public API for RL
        ...
    
    def get_agent_batch_shape(self) -> list[tuple[int]]:
        # The batch size/shape of the agents.
        # If one agent has batch_shape and observation_shape, then the agent would receive (*batch_shape, *observation_shape)
        return [() for _ in range(self.num_agents)]
    
    @abstractmethod
    def reset(
        self,
        rng: jax.Array,
        const: EnvConstType
    ) -> tuple[EnvStateType, list[Observation]]:
        """
        Public API for resetting the environment.

        [Input]
            - rng: Random number generator.
            - const: Environment constants.

        [Output]
            - state: The initial environment state of a new episode.
            - obs_list: A list of observations for each agent.
        """
        ...
    
    @abstractmethod
    def step(
        self,
        rng: jax.Array,
        const: EnvConstType,
        state: EnvStateType,
        action: Sequence[Action]
    ) -> tuple[EnvStateType, list[Observation], list[jax.Array], list[jax.Array], dict[Any, Any]]:
        """
        The `step()` method provides a standard multi-agent reinforcement learning (MARL) environment API.

        [Input]
            - rng: Random number generator array for random events.
            - const: Environment constants that configure the step behavior.
            - state: Current state of the environment.
            - action: Action (a list of actions) to be taken by the agents.

        [Output]
            - state: The environment state of next step.
            - obs_lst: A list of observations for each agent.
            - reward_lst: A list of rewards for each agent.
            - done_lst: A list of done flags for each agent.
            - info: Additional information returned by the environment.

        The returned `obs_lst`, `reward_lst`, and `done_lst` are all lists of length `num_agents`.

        It is expected that the batch_shape of obs_lst[i] and the shape of reward_lst[i]/done_lst[i] matches self.get_agent_batch_shape()[i]
        """
        ...

EnvType = TypeVar('EnvType', bound=BaseEnv)
