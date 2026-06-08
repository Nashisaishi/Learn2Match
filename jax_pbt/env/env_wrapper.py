from typing import Any, Generic, Sequence

import jax

from .base_env import BaseEnv, EnvConstType, EnvStateType, EnvType
from .spaces import Observation, Action, ObservationSpace, ActionSpace


class EnvWrapper(BaseEnv[EnvConstType, EnvStateType], Generic[EnvConstType, EnvStateType]):
    def __init__(self, env: EnvType):
        self._env = env
    
    @property
    def default_const(self) -> EnvConstType:
        return self._env.default_const
    
    @property
    def env_observation_space(self) -> tuple[ObservationSpace, dict[str, tuple[int]]]:
        return self._env.env_observation_space

    @property
    def env_action_space(self) -> tuple[ActionSpace, dict[str, tuple[int]]]:
        return self._env.env_action_space
    
    def env_reset(
        self,
        rng: jax.Array,
        const: EnvConstType
    ) -> tuple[EnvStateType, Observation]:
        return self._env.env_reset(rng, const)
    
    def env_step(
        self,
        rng: jax.Array,
        const: EnvConstType,
        state: EnvStateType,
        action: Action
    ) -> tuple[EnvStateType, Observation, jax.Array, jax.Array, dict[Any, Any]]:
        return self._env.env_step(rng, const, state, action) 
    
    @property
    def num_agents(self) -> int:
        return self._env.num_agents
    
    def get_agent_batch_shape(self) -> list[tuple[int]]:
        return self._env.get_agent_batch_shape()

    def get_observation_space(self) -> list[ObservationSpace]:
        return self._env.get_observation_space()
    
    def get_action_space(self) -> list[ActionSpace]:
        return self._env.get_action_space()   
    
    def reset(
        self,
        rng: jax.Array,
        const: EnvConstType
    ) -> tuple[EnvStateType, list[Observation]]:
        return self._env.reset(rng, const)
    
    def step(
        self,
        rng: jax.Array,
        const: EnvConstType,
        state: EnvStateType,
        action: Sequence[Action]
    ) -> tuple[EnvStateType, list[Observation], list[jax.Array], list[jax.Array], dict[Any, Any]]:
        self._env.step(rng, const, state, action)
