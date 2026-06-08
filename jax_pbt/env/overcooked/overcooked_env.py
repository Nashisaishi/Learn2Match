from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked.overcooked import Overcooked as _Overcooked, State as _OvercookedState
import numpy as np

from ..base_env import BaseEnvConst, BaseEnvState, BaseEnv
from ..spaces import Observation, Action, ObservationSpace, ActionSpace


class OvercookedConst(BaseEnvConst):
    shaped_reward_factor: float = 0.0

class OvercookedState(BaseEnvState):
    _state: _OvercookedState

class OvercookedEnv(BaseEnv[OvercookedConst, OvercookedState]):
    def __init__(self, layout: str = 'cramped_room', max_steps: int = 400):
        from flax.core.frozen_dict import FrozenDict
        from jaxmarl.environments.overcooked.layouts import overcooked_layouts as layouts
        self.layout_dict = FrozenDict(layouts[layout])
        self._env: _Overcooked = _Overcooked(layout=self.layout_dict, max_steps=max_steps)
        self.max_steps: int = max_steps
    
    @property
    def default_const(self) -> OvercookedConst:
        return OvercookedConst(max_steps=self._env.max_steps)

    @property
    def env_observation_space(self) -> tuple[ObservationSpace, dict[str, tuple[int]]]:
        obs_space = self._env.observation_space('agent_0').shape
        assert obs_space == self._env.observation_space('agent_1').shape, "The current implementation assumes two players have the same observation space in jaxmarl.overcooked"
        return ObservationSpace({'grid_2d': obs_space}), {'grid_2d': (2,)}
    
    @property
    def env_action_space(self) -> tuple[ActionSpace, dict[str, tuple[int]]]:
        num_actions = self._env.action_space('agent_0').n
        assert num_actions == self._env.action_space('agent_1').n, "The current implementation assumes two players have the same action space in jaxmarl.overcooked"
        return ActionSpace({'all': num_actions}), {'all': (2,)}
    
    def env_reset(
        self,
        rng: jax.Array,
        const: OvercookedConst
    ) -> tuple[OvercookedState, Observation]:
        obs, _state = self._env.reset(rng)
        grid_2d = jnp.stack([obs['agent_0'], obs['agent_1']])
        state = OvercookedState(_state=_state, _step=0)
        return jax.lax.stop_gradient(state), Observation({'grid_2d': jax.lax.stop_gradient(grid_2d)})
    
    def env_step(
        self,
        rng: jax.Array,
        const: OvercookedConst,
        state: OvercookedState,
        action: Action
    ) -> tuple[OvercookedState, Observation, jax.Array, jax.Array, dict[Any, Any]]:
        # Return: [state, obs, reward, done, info]
        # For all of obs, reward, done, the shape must follows [num_agents, *]
        env_action = {
            'agent_0': action['all'][0],
            'agent_1': action['all'][1]
        }
        obs, states, rewards, dones, infos = self._env.step( # why not env_step?
            key=rng,
            state=state._state,
            actions=env_action,
        )
        grid_2d = jnp.stack([obs['agent_0'], obs['agent_1']])
        obs = Observation({'grid_2d': grid_2d})
        state = OvercookedState(_state=states, _step=state._step+1)
        reward = jnp.stack([rewards['agent_0'], rewards['agent_1']])
        shaped_reward = infos['shaped_reward']
        shaped_reward = jnp.stack([shaped_reward['agent_0'], shaped_reward['agent_1']])
        reward += const.shaped_reward_factor * shaped_reward
        done = jnp.stack([dones['agent_0'], dones['agent_1']])
        info = {'_overcooked_env_info': infos}
        return jax.lax.stop_gradient(state), jax.lax.stop_gradient(obs), jax.lax.stop_gradient(reward), jax.lax.stop_gradient(done), info
    
    @property
    def num_agents(self) -> int:
        return 2

    def get_observation_space(self) -> list[ObservationSpace]:
        return [self.env_observation_space[0] for _ in range(2)]
    
    def get_action_space(self) -> list[ActionSpace]:
        return [self.env_action_space[0] for _ in range(2)]

    def reset(
        self,
        rng: jax.Array,
        const: OvercookedConst
    ) -> tuple[OvercookedState, list[Observation]]:
        state, env_obs = self.env_reset(rng, const)
        obs_lst = [env_obs[0], env_obs[1]]
        return state, obs_lst
    
    def step(
        self,
        rng: jax.Array,
        const: OvercookedConst,
        state: OvercookedState,
        action: Sequence[Action]
    ) -> tuple[OvercookedState, list[Observation], list[jax.Array], list[jax.Array], dict[Any, Any]]:
        env_action = Action({
            'all': jnp.stack([action[0]['all'], action[1]['all']])
        })
        state, env_obs, env_reward, env_done, info = self.env_step(rng, const, state, env_action)
        obs_lst = [env_obs[0], env_obs[1]]
        reward_lst = [env_reward[0], env_reward[1]]
        done_lst = [env_done[0], env_done[1]]
        return state, obs_lst, reward_lst, done_lst, info

    def create_frame_renderer(self) -> Callable[[OvercookedState, int], np.ndarray]:
        """
        Creates a frame rendering function tailored for Overcooked game states.

        Returns:
            Callable[[OvercookedState, int], np.ndarray]: A function that takes a game state and agent view size,
                                                            and returns a rendered frame as a NumPy array.
        """
        from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
        def render_frame(state: OvercookedState, agent_view_size: int) -> np.ndarray:
            """Renders a single frame from the game state.
            Args:
                state (OvercookedState): The current state of the Overcooked game.
                agent_view_size (int): The size of the agent's view.
            Returns:
                np.ndarray: The rendered frame as a NumPy array.
            """
            TILE_SIZE_PIXELS = 32
            state = state._state
            padding = agent_view_size - 2
            # Extract the relevant portion of the maze map based on agent view size
            maze_grid = np.asarray(state.maze_map[padding:-padding, padding:-padding, :])
            # Render the maze grid into a frame
            frame = OvercookedVisualizer._render_grid(
                maze_grid,
                tile_size=TILE_SIZE_PIXELS,
                highlight_mask=None,
                agent_dir_idx=state.agent_dir_idx,
                agent_inv=state.agent_inv
            )
            return frame
        return render_frame

    def export_gif(
        self,
        state_sequence: Sequence[OvercookedState],
        filename: str,
        agent_view_size: int = 5,
        fps: int = 5
    ) -> None:
        """
        Exports a sequence of game states as an animated GIF.

        [Input]
            - state_sequence (Sequence[OvercookedState]): A sequence of Overcooked game states.
            - filename (str): The file path where the GIF will be saved.
            - agent_view_size (int, optional): The size of the agent's view. Defaults to 5.
            - fps: Frame per second
        """
        import matplotlib.animation as animation   
        import matplotlib.pyplot as plt

        # Initialize the frame rendering function
        render_frame = self.create_frame_renderer()
        # Generate frames for each state in the sequence
        frames = [render_frame(state, agent_view_size) for state in state_sequence]

        fig, ax = plt.subplots()
        ax.axis('off')
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ims = [[ax.imshow(frame, animated=True)] for frame in frames]
        ani = animation.ArtistAnimation(fig, ims, interval=1000//fps, blit=True)
        ani.save(filename, writer="imagemagick", fps=fps)
        
        plt.close(fig)
