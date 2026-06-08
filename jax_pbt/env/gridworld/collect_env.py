from enum import IntEnum
from typing import Any, Sequence

from flax.struct import PyTreeNode
import jax
import jax.numpy as jnp
from jax.lax import stop_gradient

from ..base_env import BaseEnvConst, BaseEnvState, BaseEnv
from ..spaces import Observation, Action, ObservationSpace, ActionSpace

class GridWorldConst(BaseEnvConst):
    item_type_reward: jax.Array
    item_type_vector: jax.Array

class AgentState(PyTreeNode):
    pos: jax.Array
    last_step_reward: float
    visitation_map: jax.Array

class GridWorldState(BaseEnvState):
    agent_state: AgentState
    item_type: jax.Array
    item_pos: jax.Array
    item_reward_map: jax.Array
    item_vector_map: jax.Array
    wall_map: jax.Array
    global_visitation_state_map: jax.Array

class Actions(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2 
    RIGHT = 3 
    STAY = 4

class GridWorldEnv(BaseEnv[GridWorldConst, GridWorldState]):
    def __init__(
        self, 
        num_agents: int = 1,
        num_items: int = 6,
        num_positive_types: int = 1,
        num_negative_types: int = 0,
        item_vector_dim: int = 3,
        max_steps: int = 60, 
        grid_size: int = 10,
        obstacle_density: float = 0.0,
        partial_view_range: int = -1,
        item_positive_reward: float = 1.0,
        item_negative_reward: float = -1.0,
        is_obs_last_step_r: bool = False,
        is_obs_visitation_map: bool = False,
        is_obs_all_agent_pos: bool = True,
        is_item_remain_after_visit: bool = False,
    ) -> None:
        """        
        [Param]
            num_agents: Number of players
            num_items: Total number of items on the map.
            num_positive_types: Item types with positive reward.
            num_negative_types: Item types wtth negative reward.
            item_vector_dim: Item representation vector dimension.
            max_steps: Maximum steps before the environment terminates.
            partial_view_range: Defaut: -1 for global view. Otherwise: >0 for range of view to four directions.
            grid_size: The size of the grid (grid_size x grid_size).
            obstacle_density: Ratio of number_of_obstacles / grid_world_area for obstacle generation.
            item_positive_reward: Reward value for positive-rewarded items, can be override by env_const.
            item_negative_reward: Reward value for negative-rewarded items, can be override by env_const.

            # Optional:
            is_obs_last_step_r: Whether the agent's observation includes the reward received in the previous step.
            is_obs_visitation_map: Whether the agent's observation includes the agent's visitation map.
            is_obs_all_agent_pos: Whether each agent's observation includes the positions of all agents (otherwise only self-position is visible).
            is_item_remain_after_visit: Whether collected items remain visible in the environment after being picked up.
        """
        self.n_agent = num_agents
        self.n_item = num_items
        self.n_pos_types = num_positive_types
        self.n_neg_types = num_negative_types
        self.item_dim = item_vector_dim
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.n_obstacle = int(obstacle_density * (grid_size - 2) * (grid_size - 2))

        self.view_range = partial_view_range
        self.pos_r = item_positive_reward
        self.neg_r = item_negative_reward

        self.is_obs_last_step_r = is_obs_last_step_r
        self.is_obs_visitation_map = is_obs_visitation_map
        self.is_obs_all_agent_pos = is_obs_all_agent_pos
        self.is_item_remain_after_visit = is_item_remain_after_visit

        h, w = self.grid_size, self.grid_size
        self.available_positions = jnp.array([
            (x, y) for x in range(1, h - 1) for y in range(1, w - 1)
        ])
        self.obs_size = 2 * self.view_range + 1 if self.view_range > 0 else self.grid_size
    
    @property
    def default_const(self) -> GridWorldConst:
        rng = jax.random.key(0)
        r = jnp.array([self.pos_r for _ in range(self.n_pos_types)] + [self.neg_r for _ in range(self.n_neg_types)], dtype=jnp.float32)
        x = jax.random.normal(rng, (self.n_pos_types + self.n_neg_types, self.item_dim))
        x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
        return GridWorldConst(max_steps=self.max_steps, item_type_reward=r, item_type_vector=x)
    
    @property
    def env_observation_space(self) -> tuple[ObservationSpace, dict[str, tuple[int]]]:
        # 2D map observation
        grid_num_channels = {
            'map': 1,
            'item_vector_map': self.item_dim,
            'my_pos': 1,
            **({'visitation_map': 1} if self.is_obs_visitation_map else {}),
            **({'all_agent_pos': self.n_agent} if self.is_obs_all_agent_pos else {})
        }
        grid_obs_space = (self.obs_size, self.obs_size, sum(grid_num_channels.values()))

        # Vectorized observation
        agent_info_num_features = {
            'time_duration': 1,
            **({'last_reward': 1} if self.is_obs_last_step_r else {}),
        }
        agent_info_obs_space = (sum(agent_info_num_features.values()),)
        return ObservationSpace({'grid': grid_obs_space, 'agent_info': agent_info_obs_space}), {'grid': (self.n_agent,), 'agent_info': (self.n_agent,)}
    
    @property
    def env_action_space(self) -> tuple[ActionSpace, dict[str, tuple[int]]]:
        num_actions = len(Actions)
        return ActionSpace({'all': num_actions}), {'all': (self.n_agent,)}

    def _reset_positions(self, rng: jax.Array, const: GridWorldConst) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Reset the map, and the positions of all agents and items
        """
        h, w = self.grid_size, self.grid_size
        n_agent, n_item, n_obstacle = self.n_agent, self.n_item, self.n_obstacle
        available_positions = self.available_positions

        rng, rng_index = jax.random.split(rng)
        indices = jax.random.choice(rng_index, len(available_positions), shape=(n_agent + n_item + n_obstacle,), replace=False)
        agent_indicies, item_indicies, obstacle_indicies = indices[:n_agent], indices[n_agent:n_agent+n_item], indices[n_agent+n_item:]
        
        agent_pos = available_positions[agent_indicies]
        item_pos = available_positions[item_indicies]
        obstacle_pos = available_positions[obstacle_indicies]
        return agent_pos, item_pos, obstacle_pos

    def env_reset(
        self,
        rng: jax.Array,
        const: GridWorldConst
    ) -> tuple[GridWorldState, Observation]:
        """Resets the environment to an initial state."""
        h, w = self.grid_size, self.grid_size

        rng, rng_reset_pos = jax.random.split(rng)
        agent_pos, item_pos, obstacle_pos = self._reset_positions(rng_reset_pos, const)

        rng, rng_item_type = jax.random.split(rng)
        item_type = jax.random.randint(rng_item_type, (self.n_item,), 0, self.n_pos_types + self.n_neg_types)        
        item_vector = const.item_type_vector[item_type]
        item_vector_map = jnp.zeros((h, w, self.item_dim), dtype=jnp.float32)
        item_vector_map = item_vector_map.at[item_pos[:, 0], item_pos[:, 1]].set(item_vector)
        item_reward = const.item_type_reward[item_type]
        item_reward_map = jnp.zeros((h, w), dtype=jnp.float32)
        item_reward_map = item_reward_map.at[item_pos[:, 0], item_pos[:, 1]].set(item_reward)
        
        wall_map = jnp.zeros((h, w), dtype=jnp.bool)
        wall_map = wall_map.at[0, :].set(True)
        wall_map = wall_map.at[-1, :].set(True)
        wall_map = wall_map.at[:, 0].set(True)
        wall_map = wall_map.at[:, -1].set(True)
        wall_map = wall_map.at[obstacle_pos[:, 0], obstacle_pos[:, 1]].set(True)

        state = GridWorldState(
            _step=0,
            agent_state=AgentState(
                pos=agent_pos,
                last_step_reward=jnp.zeros((self.n_agent,), dtype=jnp.float32),
                visitation_map=jnp.zeros((self.n_agent, h, w), dtype=jnp.bool)
            ),
            item_type=item_type,
            item_pos=item_pos,
            item_reward_map=item_reward_map,
            item_vector_map=item_vector_map,
            wall_map=wall_map,
            global_visitation_state_map=jnp.zeros((h, w), dtype=jnp.bool)
        )
        obs = self.get_obs(state)
        return stop_gradient(state), stop_gradient(obs)

    def _crop_map(self, full_map: jax.Array, pos: jax.Array) -> jax.Array:
        full_map = jnp.pad(full_map, self.view_range)
        return jax.lax.dynamic_slice(full_map, pos, (self.obs_size, self.obs_size))
    
    def get_agent_obs(self, state: GridWorldState, agent_state: AgentState, agent_id: int) -> jax.Array:
        h, w = self.grid_size, self.grid_size
        agent_pos = state.agent_state.pos

        my_pos_map = jnp.zeros((h, w), dtype=jnp.float32).at[agent_pos[0], agent_pos[1]].set(1)
        all_pos_map = jnp.zeros((h, w, self.n_agent), dtype=jnp.float32).at[state.agent_state.pos[:, 0], state.agent_state.pos[:, 1], jnp.arange(self.n_agent)].set(1)

        # 2D observation: [H, W, Channel]
        grid_channels = {
            'map': state.wall_map.reshape(h, w, 1),
            'item_vector_map': state.item_vector_map if self.is_item_remain_after_visit else state.item_vector_map * (1 - state.global_visitation_state_map).reshape(h, w, 1),
            'my_pos': jnp.zeros((h, w, 1), dtype=jnp.float32).at[agent_pos[0], agent_pos[1], 0].set(1),
            **({'visitation_map': agent_state.visitation_map.reshape(h, w, 1)} if self.is_obs_visitation_map else {}),
            **({'all_agent_pos': jnp.zeros((h, w, self.n_agent), dtype=jnp.float32).at[state.agent_state.pos[:, 0], state.agent_state.pos[:, 1], jnp.arange(self.n_agent)].set(1)} if self.is_obs_all_agent_pos else {})
        }
        grid_obs = jnp.concatenate([v for k, v in sorted(grid_channels.items())], axis=-1)
        if self.view_range > 0:
            grid_obs = jax.vmap(self._crop_map, in_axes=(-1, None), out_axes=-1)(grid_obs, agent_pos)

        # Vectorized observation
        agent_info_features = {
            'time_duration': jnp.array([state._step / self.max_steps], dtype=jnp.float32),
            **({'last_reward': jnp.array([agent_state.last_step_reward])} if self.is_obs_last_step_r else {}),
        }
        agent_info_obs = jnp.concatenate([v for k, v in sorted(agent_info_features.items())], axis=-1)

        return Observation({'grid': grid_obs, 'agent_info': agent_info_obs})

    def get_obs(self, state: GridWorldState) -> Observation:
        return jax.vmap(self.get_agent_obs, in_axes=(None, 0, 0))(state, state.agent_state, jnp.arange(self.n_agent))

    def agent_step(self, rng: jax.Array, const: GridWorldConst, state: GridWorldState, agent_id: int, action: int) -> dict[str, jax.Array]:
        agent_pos = state.agent_state.pos[agent_id]

        # Process agent movements
        move = jnp.array([0, 0])
        move = jax.lax.cond(action == Actions.UP, lambda _: jnp.array([-1, 0]), lambda _: move, None)
        move = jax.lax.cond(action == Actions.DOWN, lambda _: jnp.array([1, 0]), lambda _: move, None)
        move = jax.lax.cond(action == Actions.LEFT, lambda _: jnp.array([0, -1]), lambda _: move, None)
        move = jax.lax.cond(action == Actions.RIGHT, lambda _: jnp.array([0, 1]), lambda _: move, None)
        new_pos = agent_pos + move
        collide_wall = state.wall_map[new_pos[0], new_pos[1]]
        new_pos = jax.lax.cond(collide_wall, lambda _: agent_pos, lambda _: new_pos, None)

        # Process reward
        old_vis_map = state.agent_state.visitation_map[agent_id]
        new_vis_map = old_vis_map.at[new_pos[0], new_pos[1]].set(True)
        if self.is_item_remain_after_visit:
            reward = (1 - old_vis_map[new_pos[0], new_pos[1]]) * state.item_reward_map[new_pos[0], new_pos[1]]
        else:
            reward = (1 - state.global_visitation_state_map[new_pos[0], new_pos[1]]) * state.item_reward_map[new_pos[0], new_pos[1]]
        
        return {
            'new_pos': new_pos,
            'new_vis_map': new_vis_map,
            'reward': reward
        }

    def env_step(
        self,
        rng: jax.Array,
        const: GridWorldConst,
        state: GridWorldState,
        action: Action
    ) -> tuple[GridWorldState, Observation, jax.Array, jax.Array, dict[Any, Any]]:
        env_action = action['all']
        rng, rng_agent_step_batch = jax.random.split(rng)
        rng_agent_step_batch = jax.random.split(rng_agent_step_batch, self.n_agent)
        data = jax.vmap(self.agent_step, in_axes=(0, None, None, 0, 0))(rng_agent_step_batch, const, state, jnp.arange(self.n_agent), env_action)

        reward = data['reward']

        num_steps = state._step + 1
        done = num_steps >= const.max_steps
        new_state = state.replace(
            _step=state._step + 1,
            agent_state=state.agent_state.replace(
                pos=data['new_pos'],
                visitation_map=data['new_vis_map'],
                last_step_reward=data['reward']
            ),
            global_visitation_state_map=jnp.any(data['new_vis_map'], axis=0)
        )

        rng, rng_reset = jax.random.split(rng)
        state_reset, obs_reset = self.env_reset(rng_reset, const)

        s, o, r, d, info = jax.lax.cond(
            done,
            lambda _: (state_reset, obs_reset, reward, jnp.repeat(done, self.n_agent), {}),
            lambda _: (new_state, self.get_obs(new_state), reward, jnp.repeat(done, self.n_agent), {}),
            None
        )
        return stop_gradient(s), stop_gradient(o), stop_gradient(r), stop_gradient(d), stop_gradient(info)
    
    @property
    def num_agents(self) -> int:
        return self.n_agent

    def get_observation_space(self) -> list[ObservationSpace]:
        return [self.env_observation_space[0] for _ in range(self.n_agent)]
    
    def get_action_space(self) -> list[ActionSpace]:
        return [self.env_action_space[0] for _ in range(self.n_agent)]

    def reset(
        self,
        rng: jax.Array,
        const: GridWorldConst
    ) -> tuple[GridWorldState, list[Observation]]:
        state, env_obs = self.env_reset(rng, const)
        obs_lst = [env_obs[i] for i in range(self.n_agent)]
        return state, obs_lst
    
    def step(
        self,
        rng: jax.Array,
        const: GridWorldConst,
        state: GridWorldState,
        action: Sequence[Action]
    ) -> tuple[GridWorldState, list[Observation], list[jax.Array], list[jax.Array], dict[Any, Any]]:
        env_action = Action({
            'all': jnp.stack([agent_action['all'] for agent_action in action])
        })
        state, env_obs, env_reward, env_done, info = self.env_step(rng, const, state, env_action)
        obs_lst = [env_obs[i] for i in range(self.n_agent)]
        reward_lst = [env_reward[i] for i in range(self.n_agent)]
        done_lst = [env_done[i] for i in range(self.n_agent)]
        return state, obs_lst, reward_lst, done_lst, info
    
    def export_gif(self, states: list[GridWorldState], filename: str = "gridworld.gif", agent_id: int = 0):
        """Creates a GIF from a list of states."""
        import matplotlib.animation as animation
        import matplotlib.patches as patches      
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns  

        fig, ax = plt.subplots(figsize=(self.grid_size / 2, self.grid_size / 2))
    
        def update(frame: int) -> None:
            ax.clear()
            state = states[frame]
    
            grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
            grid[:] = " "
            grid[state.wall_map] = "#"

            if self.view_range > 0:
                bg_map = np.array(0.3 * (1 - state.wall_map))
                for i in range(self.n_agent):
                    x, y = int(state.agent_state.pos[i][0].item()), int(state.agent_state.pos[i][1].item())
                    bg_map[max(0, x - self.view_range):min(self.grid_size, x + self.view_range + 1), max(0, y - self.view_range):min(self.grid_size, y + self.view_range + 1)] /= 0.3
            else:
                bg_map = np.array(1 - state.wall_map)

            bg_map *= (1 - state.agent_state.visitation_map[agent_id] * 0.5)

            ax.imshow(bg_map, cmap="gray", alpha=0.3)

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if grid[i, j] == " ":
                        continue
                    ax.text(j, i, grid[i, j], ha="center", va="center", fontsize=12)
            
            blue = jnp.array([0,0,1.0])
            red = jnp.array([1.0,0,0])

            for k in range(self.n_item):
                c = np.array(sns.color_palette("tab10", n_colors=10))[state.item_type[k]]
                i, j = int(state.item_pos[k, 0].item()), int(state.item_pos[k, 1].item())
                if not self.is_item_remain_after_visit and state.global_visitation_state_map[i, j]:
                    continue # this item has been collected
                square = patches.Rectangle(
                    (j - 0.5, i - 0.5),  # lower-left corner
                    1, 1,  # width and height
                    facecolor=c,
                    edgecolor=c
                )
                ax.add_patch(square)

            for i in range(self.n_agent):
                if i != agent_id and not self.is_obs_all_agent_pos:
                    continue
                s = jnp.clip(state.agent_state.last_step_reward[i], -1, 1)
                c = (red * jnp.clip(s, 0, 1) + blue * jnp.clip(-s, 0, 1)).tolist()
                square = patches.Rectangle(
                    (state.agent_state.pos[i][1] - 0.5, state.agent_state.pos[i][0] - 0.5),  # lower-left corner
                    1, 1,  # width and height
                    facecolor=c,
                    edgecolor='black'
                )
                ax.add_patch(square)
                ax.text(state.agent_state.pos[i][1], state.agent_state.pos[i][0], f"{i}", ha="center", va="center", fontsize=14, color="white")

            ax.set_xticks([])
            ax.set_yticks([])

        ani = animation.FuncAnimation(fig, update, frames=len(states), interval=500)
        ani.save(filename, writer="imagemagick")
        plt.close(fig)
