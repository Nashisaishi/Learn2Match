from typing import Any, Generic, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..utils import pytree_repeat_stack
from .base_env import BaseEnv, EnvConstType, EnvStateType, EnvType
from .spaces import Observation, Action, ObservationSpace, ActionSpace


class BatchedEnv(BaseEnv[EnvConstType, EnvStateType], Generic[EnvConstType, EnvStateType]):
    def __init__(self, env: EnvType, num_envs: int) -> None:
        """
        A parallelized wrapper for environments inheriting from BaseEnv.

        [Input]
            - env: A single-environment instance to be parallelized.
            - num_envs: Number of parallel environment copies.
        """
        self._env = env
        self.num_envs = num_envs
    
    @property
    def default_const(self) -> EnvConstType:
        return pytree_repeat_stack(self._env.default_const, (self.num_envs,))

    @property
    def env_observation_space(self) -> tuple[ObservationSpace, dict[str, tuple[int]]]:
        observation_space, observation_attr_batch_shape = self._env.env_observation_space
        observation_attr_batch_shape = {
            k: (self.num_envs, *v)
            for k, v in observation_attr_batch_shape.items()
        }
        return observation_space, observation_attr_batch_shape
    
    @property
    def env_action_space(self) -> tuple[ActionSpace, dict[str, tuple[int]]]:
        action_space, action_attr_batch_shape = self._env.env_action_space
        action_attr_batch_shape = {
            k: (self.num_envs, *v)
            for k, v in action_attr_batch_shape.items()
        }
        return action_space, action_attr_batch_shape
    
    def env_reset(
        self,
        rng: jax.Array,
        const: EnvConstType
    ) -> tuple[EnvStateType, Observation]:
        rng = jax.random.split(rng, self.num_envs)
        return jax.vmap(self._env.env_reset)(rng, const)
    
    def env_step(
        self,
        rng: jax.Array,
        const: EnvConstType,
        state: EnvStateType,
        action: Action
    ) -> tuple[EnvStateType, Observation, jax.Array, jax.Array, dict[Any, Any]]:
        rng = jax.random.split(rng, self.num_envs)
        return jax.vmap(self._env.env_step)(rng, const, state, action) 
    
    @property
    def num_agents(self) -> int:
        return self._env.num_agents

    def get_observation_space(self) -> list[ObservationSpace]:
        return self._env.get_observation_space()
    
    def get_action_space(self) -> list[ActionSpace]:
        return self._env.get_action_space()   
    
    def get_agent_batch_shape(self) -> list[tuple[int]]:
        return [(self.num_envs, *batch_shape) for batch_shape in self._env.get_agent_batch_shape()]
    
    def reset(
        self,
        rng: jax.Array,
        const: EnvConstType
    ) -> tuple[EnvStateType, list[Observation]]:
        rng = jax.random.split(rng, self.num_envs)
        return jax.vmap(self._env.reset)(rng, const)
    
    def step(
        self,
        rng: jax.Array,
        const: EnvConstType,
        state: EnvStateType,
        action: Sequence[Action]
    ) -> tuple[EnvStateType, list[Observation], list[jax.Array], list[jax.Array], dict[Any, Any]]:
        rng = jax.random.split(rng, self.num_envs)
        return jax.vmap(self._env.step)(rng, const, state, action)

class RoleAssignmentBatchedEnv(BatchedEnv[EnvConstType, EnvStateType], Generic[EnvConstType, EnvStateType]):
    def __init__(self, env: EnvType, num_envs: int, assignments: Sequence[Sequence[int]]) -> None:
        """
        A batched environment wrapper that groups the `num_agents` of a single environment into `M` *roles* (e.g., RL policies).

        [Input]
            - env: A single-environment instance to be parallelized.
            - num_envs: Number of parallel environment copies.
            - assignments: A list of agent idx groups. Each group defines one role.
            Example: assignments = [[0,1,2], [3,4,5]]
                -> The batched environment has M = 2 roles.
                -> Role 0 controls agents 0,1,2; Role 1 controls agents 3,4,5.
                -> For each role, observations/actions/rewards/dones for its agents are batch-flattened ((num_envs, *agent_batch_shape) -> (batch_size,)) and concatenated along the axis-0 (batch_size dimension). See step().
        """
        self._env = env
        self.single_env_num_agents = len(self._env.get_agent_batch_shape())
        self.single_env_observation_spaces = self._env.get_observation_space()
        self.single_env_action_spaces = self._env.get_action_space()
        
        self.num_envs = num_envs
        self.assignments: list[list[int]] = [sorted(list(group)) for group in assignments]
        self.agent_roles: list[int] = [None for _ in range(self.single_env_num_agents)]
        for role_id, role in enumerate(self.assignments):
            if len(role) == 0:
                raise ValueError(f"Role {role_id} is empty. Each role must contain at least one agent.")
            for i in role:
                if self.agent_roles[i] is not None:
                    raise ValueError(
                        f"Agent {i} is assigned more than once. "
                        f"Full assignments: {assignments}"
                    )
                self.agent_roles[i] = role_id
        _missing_agents = [i for i, role_id in enumerate(self.agent_roles) if role_id is None]
        if _missing_agents:
            raise ValueError(
                f"The following agents are not assigned to any role: {_missing_agents}. "
                f"Full assignments: {assignments}"
            )

        self.single_env_agent_batch_shape = self._env.get_agent_batch_shape()
        self.per_role_split_section, self.per_role_count = [], []
        for role in self.assignments:
            sizes = [int(np.prod(self.single_env_agent_batch_shape[i])) * self.num_envs for i in role]
            cum = np.cumsum(sizes)
            self.per_role_split_section.append(list(cum[:-1]))
            self.per_role_count.append(cum[-1])

        self._role_observation_spaces = []
        self._role_action_spaces = []
        for role_idx, role in enumerate(self.assignments):
            first_observation = self.single_env_observation_spaces[role[0]]
            first_action = self.single_env_action_spaces[role[0]]
            for agent_idx in role[1:]:
                if self.single_env_observation_spaces[agent_idx] != first_observation:
                    raise ValueError(
                        f"All agents assigned to role {role_idx} must share the same ObservationSpace. "
                        f"Mismatch between agent {role[0]} and {agent_idx}."
                    )
                if self.single_env_action_spaces[agent_idx] != first_action:
                    raise ValueError(
                        f"All agents assigned to role {role_idx} must share the same ActionSpace. "
                        f"Mismatch between agent {role[0]} and {agent_idx}."
                    )
            self._role_observation_spaces.append(first_observation)
            self._role_action_spaces.append(first_action)

    @property
    def num_agents(self) -> int:
        return len(self.assignments)

    def get_observation_space(self) -> list[ObservationSpace]:
        return self._role_observation_spaces
    
    def get_action_space(self) -> list[ActionSpace]:
        return self._role_action_spaces
    
    def get_agent_batch_shape(self) -> list[tuple[int]]:
        return [(role_count,) for role_count in self.per_role_count]
    
    def _assign_obs(self, _env_obs_lst: list[Observation]) -> list[Observation]:
        """
        Assign internal-environment observations to roles.

        Each agent's observation (num_envs, *agent_batch_shape) is flattened to (num_envs * batch_size,) and then all agents in the same role are concatenated along axis-0.

        Example:
            Suppose the environment has 3 agents:
                agent_1: (*batch_1, obs_shape_A)
                agent_2: (*batch_2, obs_shape_A)
                agent_3: (*batch_3, obs_shape_B)
            And two roles: A = [1,2], B = [3].
            Then:
                role_A: ((batch_1 + batch_2) * num_envs, obs_shape_A)
                role_B: (batch_3 * num_envs, obs_shape_B)
        """
        return [
            Observation.concatenate([
                Observation.batch_flatten(
                    _env_obs_lst[i],
                    batch_shape=(self.num_envs, *self.single_env_agent_batch_shape[i])
                )
                for i in role
            ], axis=0)
            for role in self.assignments
        ]

    def reset(
        self,
        rng: jax.Array,
        const: EnvConstType
    ) -> tuple[EnvStateType, list[Observation]]:
        rng = jax.random.split(rng, self.num_envs)
        state, _env_obs_lst = jax.vmap(self._env.reset)(rng, const)
        role_obs_lst = self._assign_obs(_env_obs_lst)
        return state, role_obs_lst
    
    def step(
        self,
        rng: jax.Array,
        const: EnvConstType,
        state: EnvStateType,
        action: Sequence[Action]
    ) -> tuple[EnvStateType, list[Observation], list[jax.Array], list[jax.Array], dict[Any, Any]]:
        rng = jax.random.split(rng, self.num_envs)
        per_role_env_action = [
            Action.split(role_action, split_section, axis=0)
            for role_action, split_section in zip(action, self.per_role_split_section)
        ]
        _env_action_lst = [
            per_role_env_action[agent_role][self.assignments[agent_role].index(i)]
            for i, agent_role in enumerate(self.agent_roles)
        ]
        _env_action_lst = [
            Action.batch_unflatten(
                agent_action,
                batch_shape=(self.num_envs, *self.single_env_agent_batch_shape[i])
            )
            for i, agent_action in enumerate(_env_action_lst)
        ]
        state, _env_obs_lst, _env_reward_lst, _env_done_lst, info = jax.vmap(self._env.step)(rng, const, state, _env_action_lst)
        role_obs_lst = self._assign_obs(_env_obs_lst)
        role_reward_lst = [jnp.concatenate([_env_reward_lst[i].flatten() for i in role]) for role in self.assignments]
        role_done_lst = [jnp.concatenate([_env_done_lst[i].flatten() for i in role]) for role in self.assignments]
        return state, role_obs_lst, role_reward_lst, role_done_lst, info


class HomogeneousBatchedEnv(RoleAssignmentBatchedEnv[EnvConstType, EnvStateType], Generic[EnvConstType, EnvStateType]):
    def __init__(self, env: EnvType, num_envs: int) -> None:
        """
        Run `num_envs` copies of the same environment and treat *all internal agents* as a single homogeneous role. All agents must share identical observation and action spaces.

        Example (2 homogeneous internal agents):
            Suppose agent_0 has batch_shape = (2,), agent_1 has batch_shape = (3, 5), and both have obs_shape = (XX, YY).
            Then for num_envs = 4:
                agent_0 obs: (4, 2, XX, YY) -> flattened to (8, XX, YY)
                agent_1 obs: (4, 3, 5, XX, YY) -> flattened to (60, XX, YY)
            Combined role observation:
                concatenate on axis 0 -> (68, XX, YY)

            Rewards/dones are processed the same way.

        [Input]
            - env: A single-environment instance to be replicated.
            - num_envs: Number of parallel environment copies.
        """
        assignments = [list(range(env.num_agents))]
        super().__init__(env, num_envs, assignments)


class DynamicFixedSizeAssignmentEnv(BatchedEnv[EnvConstType, EnvStateType], Generic[EnvConstType, EnvStateType]):
    """
    A batched environment wrapper that supports dynamic mapping of agent data (observations/rewards/dones) 
    to external roles. Specifically, it maps each **internal agent's** parallel environment samples to external roles.

    The assignment defines which samples from the internal `num_agents, num_envs` pool are gathered into 
    each role's buffer at every step. `use_batch_indexing` is an optional feature that provides more fine-grained 
    control over each internal agent's flattened `batch_size` dimension.

    The user supplies, at each `reset` / `step` call, a list of per-role assignments:
        - per_role_per_agent_env_idx: list[list[jax.Array]]
            - role #i receives data from agent #j on environment indices `per_role_per_agent_env_idx[i][j]`.
        - per_role_per_agent_batch_idx: list[list[jax.Array]] (optional, used if `use_batch_indexing=True`)
            - role #i receives data from agent #j on specific [env[i][j], batch[i][j]] indices of the flattened data of agent #j, e.g., obs_lst[j].flatten(num_envs, prod(batch_shape), *observation_shape)
    """
    def __init__(self, env: EnvType, num_envs: int, num_roles: int, use_batch_indexing: bool = False) -> None:
        """
        [Input]
            - env: Internal environment to be batched.
            - num_envs: Number of environment copies.
        
        [Param]
            - single_env_num_agents: Number of agents in the single environment.
            - single_env_observation_spaces: List of observation spaces for each internal agent.
            - single_env_action_spaces: List of action spaces for each internal agent.
            - single_env_agent_batch_shape: List of batch shapes for each internal agent.
            - single_env_agent_batch_size: Flattened batch size for each internal agent.
            - num_roles: Number of external roles to assign data to.
            - use_batch_indexing: If True, allows fine-grained idx selection within agent batches.
        """
        self._env = env
        self.num_envs = num_envs
        self.single_env_num_agents = self._env.num_agents
        self.single_env_observation_spaces = self._env.get_observation_space()
        self.single_env_action_spaces = self._env.get_action_space()
        self.single_env_agent_batch_shape = self._env.get_agent_batch_shape()
        self.single_env_agent_batch_size = [int(np.prod(s)) for s in self.single_env_agent_batch_shape]
        self.num_roles = num_roles
        self.use_batch_indexing = use_batch_indexing

    @property
    def num_agents(self) -> int:
        return self.num_roles

    def get_observation_space(self) -> list[ObservationSpace]:
        raise NotImplementedError("Dynamic assignment: observation space depends on the provided indices. Caller must override this or define space externally.")

    def get_action_space(self) -> list[ActionSpace]:
        raise NotImplementedError("Dynamic assignment: action space depends on the provided indices. Caller must override this or define space externally.")

    def get_agent_batch_shape(self) -> list[tuple[int]]:
        raise NotImplementedError("Dynamic assignment: batch shapes are not fixed per role. Caller must override this or define agent_batch_shape externally.")
    
    def _data_assignment(
        self,
        per_agent_all_env_data_lst: Sequence[jax.Array | Observation],
        per_role_per_agent_env_idx: Sequence[Sequence[jax.Array]],
        per_role_per_agent_batch_idx: Sequence[Sequence[jax.Array]] = None
    ) -> list[jax.Array | Observation]:
        """
        Gather internal agent observation/reward/done into role-specific lists based on indices.
        
        Let n = _env.num_agents, m = num_roles.
        [Input]
            - per_agent_all_env_data_lst: [(num_envs, *agent_batch_shape, *data_shape) 1...n]
            - per_role_per_agent_env_idx: [agent_data[i][env_idx], i = 1...n] -> role_data[j]
            - per_role_per_agent_batch_idx: (optional) [agent_data[i].flatten()[env_idx, batch_idx], i = 1...n] -> role_data[j]
            
        [Output]
            - per_role_data_lst: [(role_batch_size, *data_shape) 1...m]
        """
        def _flatten(data: jax.Array | Observation, batch_shape: tuple[int]) -> jax.Array | Observation:
            if isinstance(data, Observation):
                return Observation.batch_flatten(data, batch_shape)
            else:
                return data.flatten()
        
        def _cat(data_lst: Sequence[jax.Array | Observation]) -> jax.Array | Observation:
            return Observation.concatenate(data_lst, axis=0) if isinstance(data_lst[0], Observation) else jnp.concatenate(data_lst, axis=0)
        
        if not self.use_batch_indexing:
            return [
                _cat([
                    _flatten(agent_all_env_data[env_idx], (len(env_idx), *batch_shape))
                    for agent_all_env_data, env_idx, batch_shape in zip(per_agent_all_env_data_lst, per_agent_env_idx, self.single_env_agent_batch_shape)
                    if len(env_idx) > 0
                ])
                for per_agent_env_idx in per_role_per_agent_env_idx
            ]
        else:
            return [
                _cat([
                    jax.vmap(_flatten, in_axes=(0, None))(agent_all_env_data, batch_shape)[env_idx, batch_idx]
                    for agent_all_env_data, env_idx, batch_idx, batch_shape in zip(per_agent_all_env_data_lst, role_per_agent_env_idx, per_role_agent_batch_idx, self.single_env_agent_batch_shape)
                    if len(env_idx) > 0
                ])
                for role_per_agent_env_idx, per_role_agent_batch_idx in zip(per_role_per_agent_env_idx, per_role_per_agent_batch_idx)
            ]
    
    def _map_actions_to_agents(
        self,
        per_role_action_lst: Sequence[Action],
        per_role_per_agent_env_idx: Sequence[Sequence[jax.Array]],
        per_role_per_agent_batch_idx: Sequence[Sequence[jax.Array]] = None
    ) -> list[Action]:
        """
        Map external role actions back to their corresponding internal agents based on provided indices.
        
        Let n = _env.num_agents, m = num_roles.
        [Input]
            - per_role_action_lst: [(role_batch_size, *action_shape) 1...m]
            - per_role_per_agent_env_idx: role_action[j].split()[i] -> agent_action[i][env_idx] 
            - per_role_per_agent_batch_idx: (optional) role_action[j].split()[i] -> agent_action[i].flatten()[env_idx, batch_idx]
            
        [Output]
            - per_agent_action_lst: [(num_envs, *agent_batch_shape, *action_shape) 1...n]
        """
        # Dynamically calculate split sections for each role action based on the size of the assignment
        per_role_per_agent_num_assignments = [
            [
                len(env_idx) * batch_size if not self.use_batch_indexing else len(env_idx)
                for env_idx, batch_size in zip(per_agent_env_idx, self.single_env_agent_batch_size)
            ]
            for per_agent_env_idx in per_role_per_agent_env_idx
        ]
        per_role_per_agent_action = [
            Action.split(role_action, np.cumsum(per_agent_num_assignments)[:-1], axis=0)
            for role_action, per_agent_num_assignments in zip(per_role_action_lst, per_role_per_agent_num_assignments)
        ] # [[(role_to_agent_batch_size, *action_space) 1...n] 1...m]
        
        # Scatter the role-specific actions into the correct agent-wise action buffers using assigned indices.
        if not self.use_batch_indexing:
            per_agent_action_lst = [
                self.single_env_action_spaces[a].example(batch_shape=(self.num_envs, *self.single_env_agent_batch_shape[a])).at[
                    jnp.concatenate([
                        per_agent_env_idx[a] for per_agent_env_idx in per_role_per_agent_env_idx if len(per_agent_env_idx[a]) > 0
                    ], axis=0)
                ].set(
                    Action.concatenate([
                        Action.batch_unflatten(all_agent_action[a], batch_shape=(len(all_agent_env_idx[a]), *self.single_env_agent_batch_shape[a]))
                        for all_agent_env_idx, all_agent_action in zip(per_role_per_agent_env_idx, per_role_per_agent_action)
                        if len(all_agent_env_idx[a]) > 0
                    ], axis=0)
                ) # (num_envs, *agent_batch_shape, *action_shape)
                for a in range(self.single_env_num_agents)
            ]
        else:
            per_agent_action_flatten_lst = [
                self.single_env_action_spaces[a].example(batch_shape=(self.num_envs, self.single_env_agent_batch_size[a])).at[
                    jnp.concatenate([
                        all_agent_env_idx[a] for all_agent_env_idx in per_role_per_agent_env_idx if len(all_agent_env_idx[a]) > 0
                    ], axis=0),
                    jnp.concatenate([
                        all_agent_batch_idx[a] for all_agent_batch_idx in per_role_per_agent_batch_idx if len(all_agent_batch_idx[a]) > 0
                    ], axis=0)
                ].set(
                    Action.concatenate([
                        all_agent_action[a]
                        for all_agent_env_idx, all_agent_action in zip(per_role_per_agent_env_idx, per_role_per_agent_action)
                        if len(all_agent_env_idx[a]) > 0
                    ], axis=0)
                ) # (num_envs, agent_batch_size, *action_shape)
                for a in range(self.single_env_num_agents)
            ]
            per_agent_action_lst = [
                jax.vmap(Action.batch_unflatten, in_axes=(0, None))(action_flatten, batch_shape) # (num_envs, *agent_batch_shape, *action_shape)
                for action_flatten, batch_shape in zip(per_agent_action_flatten_lst, self.single_env_agent_batch_shape)
            ]

        return per_agent_action_lst
    
    def reset(
        self,
        rng: jax.Array,
        const: EnvConstType,
        per_role_per_agent_env_idx: Sequence[Sequence[jax.Array]],
        per_role_per_agent_batch_idx: Sequence[Sequence[jax.Array]] = None
    ) -> tuple[EnvStateType, list[Observation]]:
        env_state, all_obs = super().reset(rng, const)
        obs_lst = self._data_assignment(all_obs, per_role_per_agent_env_idx, per_role_per_agent_batch_idx)
        return env_state, obs_lst

    def step(
        self,
        rng: jax.Array,
        const: EnvConstType,
        state: EnvStateType,
        action: Sequence[Action],
        per_role_per_agent_env_idx: Sequence[Sequence[jax.Array]],
        per_role_per_agent_batch_idx: Sequence[Sequence[jax.Array]] = None
    ) -> tuple[EnvStateType, list[Observation], list[jax.Array], list[jax.Array], dict[Any, Any]]:
        all_action = self._map_actions_to_agents(action, per_role_per_agent_env_idx, per_role_per_agent_batch_idx)        
        env_state, all_obs, all_reward, all_done, info = super().step(rng, const, state, all_action)
        obs_lst = self._data_assignment(all_obs, per_role_per_agent_env_idx, per_role_per_agent_batch_idx)
        reward_lst = self._data_assignment(all_reward, per_role_per_agent_env_idx, per_role_per_agent_batch_idx)
        done_lst = self._data_assignment(all_done, per_role_per_agent_env_idx, per_role_per_agent_batch_idx)
        return env_state, obs_lst, reward_lst, done_lst, info


class FrozenAssignmentEnv(DynamicFixedSizeAssignmentEnv[EnvConstType, EnvStateType], Generic[EnvConstType, EnvStateType]):
    """
    A batched environment with static role assignments. Assignments can be defined in four ways:
        1. env_agent_to_role: Map each (environment, agent) pair to a role ID.
        2. env_agent_batch_to_role: Map fine-grained batch samples (environment, agent, batch-id) to roles.
        3. per_role_per_agent_env_idx: Explicitly provide environment indices for each agent per role.
        4. per_role_per_agent_batch_idx: Explicitly provide fine-grained environment and batch indices.
    """
    def __init__(
        self,
        env: BaseEnv,
        num_envs: int,
        env_agent_to_role: Sequence[Sequence[int]] = None,
        env_agent_batch_to_role: Sequence[Sequence[Sequence[int]]] = None,
        per_role_per_agent_env_idx: Sequence[Sequence[jax.Array]] = None,
        per_role_per_agent_batch_idx: Sequence[Sequence[jax.Array]] = None 
    ) -> None:
        """
        [Input]
            - env: Internal environment to be batched.
            - num_envs: Number of environment copies.
        
        [Param]
            - single_env_num_agents: Number of agents in the single environment.
            - single_env_observation_spaces: List of observation spaces for each internal agent.
            - single_env_action_spaces: List of action spaces for each internal agent.
            - single_env_agent_batch_shape: List of batch shapes for each internal agent.
            - single_env_agent_batch_size: Flattened batch size for each internal agent.
            - num_roles: Number of external roles to assign data to.
            - use_batch_indexing: If True, allows fine-grained idx selection within agent batches.
            - per_role_count: Total number of samples assigned to each role (after flattening env and agent batch dimensions).
        """
        self._env = env
        self.num_envs = num_envs
        self.single_env_num_agents = self._env.num_agents
        self.single_env_observation_spaces = self._env.get_observation_space()
        self.single_env_action_spaces = self._env.get_action_space()
        self.single_env_agent_batch_shape = self._env.get_agent_batch_shape()
        self.single_env_agent_batch_size = [int(np.prod(s)) for s in self.single_env_agent_batch_shape]

        if env_agent_to_role is not None:
            self.num_roles = int(np.max(env_agent_to_role)) + 1
            self.use_batch_indexing = False
            self.per_role_count = [0 for _ in range(self.num_roles)]
            self.per_role_per_agent_env_idx = [[[] for _ in range(self.single_env_num_agents)] for __ in range(self.num_roles)]
            for e in range(self.num_envs):
                for a in range(self.single_env_num_agents):
                    role = env_agent_to_role[e][a]
                    self.per_role_count[role] += self.single_env_agent_batch_size[a]
                    self.per_role_per_agent_env_idx[role][a].append(e)
            for role in range(self.num_roles):
                for a in range(self.single_env_num_agents):
                    self.per_role_per_agent_env_idx[role][a] = jnp.array(self.per_role_per_agent_env_idx[role][a], dtype=jnp.int32)
            self.per_role_per_agent_batch_idx = None

        elif env_agent_batch_to_role is not None:
            self.num_roles = int(np.max([[np.max(_l2) for _l2 in _l1] for _l1 in env_agent_batch_to_role])) + 1
            self.use_batch_indexing = True
            self.per_role_count = [0 for _ in range(self.num_roles)]
            self.per_role_per_agent_env_idx = [[None for _ in range(self.single_env_num_agents)] for __ in range(self.num_roles)]
            self.per_role_per_agent_batch_idx = [[None for _ in range(self.single_env_num_agents)] for __ in range(self.num_roles)] 

            for a in range(self.single_env_num_agents):
                agent_a_role_map = np.array([env_agent_batch_to_role[e][a] for e in range(self.num_envs)]) # [env, batch-id] -> role
                batch_size_a = self.single_env_agent_batch_size[a]
                env_grid, batch_grid = np.mgrid[:self.num_envs, :batch_size_a]
                for role in range(self.num_roles):
                    mask = (agent_a_role_map == role)
                    e_idx = env_grid[mask]
                    b_idx = batch_grid[mask]
                    self.per_role_count[role] += len(e_idx)
                    self.per_role_per_agent_env_idx[role][a] = jnp.array(e_idx, dtype=jnp.int32)
                    self.per_role_per_agent_batch_idx[role][a] = jnp.array(b_idx, dtype=jnp.int32)

        elif per_role_per_agent_env_idx is not None and per_role_per_agent_batch_idx is None:
            self.per_role_per_agent_env_idx = [list(l) for l in per_role_per_agent_env_idx]
            self.num_roles = len(per_role_per_agent_env_idx)
            self.use_batch_indexing = False
            self.per_role_count = [
                sum([
                    len(agent_env_indices) * agent_batch_size
                    for agent_env_indices, agent_batch_size in zip(role_per_agent_env_idx, self.single_env_agent_batch_size)
                ])
                for role_per_agent_env_idx in self.per_role_per_agent_env_idx
            ]
            self.per_role_per_agent_batch_idx = None

        elif per_role_per_agent_batch_idx is not None:
            self.per_role_per_agent_env_idx = [list(l) for l in per_role_per_agent_env_idx]
            self.per_role_per_agent_batch_idx = [list(l) for l in per_role_per_agent_batch_idx]
            self.num_roles = len(per_role_per_agent_env_idx)
            self.use_batch_indexing = True
            self.per_role_count = [
                sum([len(agent_env_indices) for agent_env_indices in role_per_agent_env_idx])
                for role_per_agent_env_idx in self.per_role_per_agent_env_idx
            ]
        else:
            raise ValueError("At least one of the four assignment methods (env_agent_to_role, env_agent_batch_to_role, per_role_per_agent_env_idx, or per_role_per_agent_batch_idx) must be provided.")
        
        self._role_observation_spaces = []
        self._role_action_spaces = []
        for role_idx in range(self.num_roles):
            assigned_agents = [a for a, idx_list in enumerate(self.per_role_per_agent_env_idx[role_idx]) if len(idx_list) > 0]
            if not assigned_agents:
                raise ValueError(f"Role {role_idx} must be assigned at least one agent sample.")

            first_agent = assigned_agents[0]
            ref_obs = self.single_env_observation_spaces[first_agent]
            ref_act = self.single_env_action_spaces[first_agent]
            for a in assigned_agents[1:]:
                if self.single_env_observation_spaces[a] != ref_obs:
                    raise ValueError(f"Space mismatch for role {role_idx}: agent {a} vs {first_agent}")
            self._role_observation_spaces.append(ref_obs)
            self._role_action_spaces.append(ref_act)

    @property
    def num_agents(self) -> int:
        return self.num_roles
    
    def get_observation_space(self) -> list[ObservationSpace]:
        return self._role_observation_spaces
    
    def get_action_space(self) -> list[ActionSpace]:
        return self._role_action_spaces
    
    def get_agent_batch_shape(self) -> list[tuple[int]]:
        return [(role_count,) for role_count in self.per_role_count]

    def reset(self, rng: jax.Array, const: EnvConstType) -> tuple[EnvStateType, list[Observation]]:
        return super().reset(
            rng=rng, const=const,
            per_role_per_agent_env_idx=self.per_role_per_agent_env_idx,
            per_role_per_agent_batch_idx=self.per_role_per_agent_batch_idx
        )
    
    def step(self, rng: jax.Array, const: EnvConstType, state: EnvStateType, action: Sequence[Action]):
        return super().step(
            rng=rng, const=const, state=state, action=action,
            per_role_per_agent_env_idx=self.per_role_per_agent_env_idx,
            per_role_per_agent_batch_idx=self.per_role_per_agent_batch_idx
        )
