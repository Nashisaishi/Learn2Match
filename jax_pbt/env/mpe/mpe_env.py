from typing import Any, Sequence

import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from jaxmarl.environments.mpe.default_params import DISCRETE_ACT, CONTINUOUS_ACT
from jaxmarl.environments.mpe.simple import State as _MPEState, SimpleMPE as _MPE
from jaxmarl.environments.spaces import Box as _Box, Discrete as _Discrete

from ..base_env import BaseEnvConst, BaseEnvState, BaseEnv
from ..spaces import Observation, Action, ObservationSpace, ActionSpace, ContinuousSpace, DiscreteSpace
from .make import make


class MPEConst(BaseEnvConst):
    pass

class MPEState(BaseEnvState):
    _state: _MPEState

class MPEEnv(BaseEnv[MPEConst, MPEState]):
    def __init__(self, scenario: str = 'MPE_simple_v3', action_type: str = 'discrete', max_steps: int = 25, clip_action: bool = False, **env_kwargs):
        """
        In JaxMARL MPE, the observations, actions, rewards, dones are all represented by dicts:
            {
                agent_name: data for that agent
            }
        """
        if action_type.lower() in ['discrete', 'dis', 'category', 'categorical']:
            self.action_type = DISCRETE_ACT
        elif action_type.lower() in ['continuous', 'con', 'box']:
            self.action_type = CONTINUOUS_ACT
        else:
            raise ValueError(f"Action type {action_type} not supported")

        self._env: _MPE = make(env_id=scenario, action_type=self.action_type, max_steps=max_steps, **env_kwargs)
        self.env_agent_names = self._env.agents
        self.max_steps: int = max_steps
        self.clip_action = clip_action
        if self.clip_action and self.action_type == CONTINUOUS_ACT:
            self.action_clip_list = [(space[CONTINUOUS_ACT].low, space[CONTINUOUS_ACT].high) for space in self.get_action_space()]
    
    def get_agent_names(self) -> list[str]:
        return self.env_agent_names.copy()

    @property
    def default_const(self) -> MPEConst:
        return MPEConst(max_steps=self.max_steps)

    @property
    def env_observation_space(self) -> tuple[ObservationSpace, dict[str, tuple[int]]]:
        obs_space = ObservationSpace({
            a: ContinuousSpace(shape=space.shape, low=space.low, high=space.high)
            for a, space in self._env.observation_spaces.items()
        })
        return obs_space, {k: () for k in obs_space.keys()}
    
    @property
    def env_action_space(self) -> tuple[ActionSpace, dict[str, tuple[int]]]:
        action_space = ObservationSpace({
            a: DiscreteSpace(n=space.n) if isinstance(space, _Discrete) else ContinuousSpace(shape=space.shape, low=space.low, high=space.high)
            for a, space in self._env.action_spaces.items()
        })
        return action_space, {k: () for k in action_space.keys()}
    
    def env_reset(
        self,
        rng: jax.Array,
        const: MPEConst
    ) -> tuple[MPEState, Observation]:
        _obs, _state = self._env.reset(rng)
        state = MPEState(_state=_state, _step=0)
        return stop_gradient(state), stop_gradient(Observation(_obs))
    
    def env_step(
        self,
        rng: jax.Array,
        const: MPEConst,
        state: MPEState,
        action: Action
    ) -> tuple[MPEState, Observation, jax.Array, jax.Array, dict[Any, Any]]:
        # Returns [state, obs, reward, done, info]
        # For reward and done, the shape shoule be (num_agents,)
        env_action = action.raw()
        _obs, _states, _rewards, _dones, _infos = self._env.step(
            key=rng,
            state=state._state,
            actions=env_action
        )
        state = MPEState(_state=_states, _step=state._step+1)
        obs = Observation(_obs)
        reward = jnp.array([_rewards[a] for a in self.env_agent_names], dtype=jnp.float32)
        done = jnp.array([_dones[a] for a in self.env_agent_names], dtype=jnp.bool)
        info = _infos.update({'done.__all__': _dones['__all__']})
        return stop_gradient(state), stop_gradient(obs), stop_gradient(reward), stop_gradient(done), stop_gradient(info)
    
    @property
    def num_agents(self) -> int:
        return len(self.env_agent_names)

    def get_observation_space(self) -> list[ObservationSpace]:
        env_observation_space, _ = self.env_observation_space
        return [ObservationSpace({'symbolic': env_observation_space[a]}) for a in self.env_agent_names]
    
    def get_action_space(self) -> list[ActionSpace]:
        env_action_space, _ = self.env_action_space
        return [ActionSpace({self.action_type: env_action_space[a]}) for a in self.env_agent_names]

    def reset(
        self,
        rng: jax.Array,
        const: MPEConst
    ) -> tuple[MPEState, list[Observation]]:
        state, env_obs = self.env_reset(rng, const)
        obs_lst = [Observation({'symbolic': env_obs[a]}) for a in self.env_agent_names]
        return state, obs_lst
    
    def step(
        self,
        rng: jax.Array,
        const: MPEConst,
        state: MPEState,
        action: Sequence[Action]
    ) -> tuple[MPEState, list[Observation], list[jax.Array], list[jax.Array], dict[Any, Any]]:
        if self.clip_action and self.action_type == CONTINUOUS_ACT:
            action = [jax.tree_util.tree_map(lambda x: jnp.clip(x, low, high), a) for a, (low, high) in zip(action, self.action_clip_list)]
        env_action = Action({
            a: action[i][self.action_type] for i, a in enumerate(self.env_agent_names)
        })
        state, env_obs, env_reward, env_done, info = self.env_step(rng, const, state, env_action)
        obs_lst = [Observation({'symbolic': env_obs[a]}) for a in self.env_agent_names]
        reward_lst = [env_reward[i] for i in range(self.num_agents)]
        done_lst = [env_done[i] for i in range(self.num_agents)]
        return state, obs_lst, reward_lst, done_lst, info

    def export_gif(
        self,
        state_sequence: Sequence[MPEState],
        filename: str,
    ) -> None:
        """Exports a sequence of game states as an animated GIF.

        Args:
            state_sequence (Sequence[MPEState]): A sequence of MPE game states.
            output_filename (str): The file path where the GIF will be saved.
        """
        from jaxmarl.environments.mpe import MPEVisualizer
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        # Changed from https://github.com/FLAIROx/JaxMARL/blob/main/jaxmarl/environments/mpe/mpe_visualizer.py 
        def animate(
            self: MPEVisualizer,
            view: bool = True,
        ) -> None:
            """Anim for 2D fct - x (#steps, #pop, 2) & fitness (#steps, #pop)"""
            ani = animation.FuncAnimation(
                self.fig,
                self.update,
                frames=len(self.state_seq),
                blit=False,
                interval=self.interval,
            )
            # Save the animation to a gif
            ani.save(filename, writer="imagemagick")

            if view:
                plt.show(block=True)
        viz = MPEVisualizer(self._env, state_seq=[state._state for state in state_sequence])
        animate(viz, view=False)
