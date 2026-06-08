from enum import IntEnum
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import seaborn as sns
import matplotlib.patches as patches

from ..base_env import BaseEnvConst, BaseEnvState, BaseEnv
from ..spaces import Observation, Action, ObservationSpace, ActionSpace

class GridWorldConst(BaseEnvConst):
    goal_detect_reward: float

class GridWorldState(BaseEnvState):
    agent_pos: jax.Array
    agent_allow_move: jax.Array
    agent_trial_best_score: jax.Array
    agent_discount_trial_ret: jax.Array
    goal_pos_map: jax.Array
    goal_score_map: jax.Array
    wall_map: jax.Array

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
        num_goals: int = 3,
        grid_size: int = 20,
        view_range: int = 4, 
        clutter_density: float = 0.0,
        trial_steps: int = 20,
        max_steps: int = 120, 
        goal_score_scale: float = 1.0,
        stop_after_goal_reach: bool = True,
        last_ret_scale: float = 0.4,
        history_ret_scale: float = 0.6
    ) -> None:
        """        
        [Param]
            num_agents: Number of players
            num_goals: Number of goals.
            grid_size: The size of the grid (grid_size x grid_size).
            clutter_density: Ratio of number_of_clutters / grid_world_size for clutter generation
            view_range: Range of view to four directions.
            trial_steps: Number of steps for each trial
            max_steps: Maximum steps before the environment terminates.
            distance_penalty: Penalty for each step based on Manhattan distance to the goal.
            goal_sore_scale: Scale of how goal scores are randomized.
            stop_after_goal_reach: Agents will stop after reaching a goal until the end of a trial.
            last_ret_scale, history_ret_scale: Determine how trial returns are cumulated as prestige
        """
        self.n_agent = num_agents
        self.n_goal = num_goals
        self.grid_size = grid_size
        self.n_clutter = int(clutter_density * (grid_size - 2) * (grid_size - 2))
        self.trial_steps = trial_steps
        self.max_steps = max_steps
        self.view_range = view_range
        self.goal_score_scale = goal_score_scale
        self.stop_after_goal_reach = stop_after_goal_reach

        self.last_ret_scale = last_ret_scale
        self.history_ret_scale = history_ret_scale

        h, w = self.grid_size, self.grid_size
        self.available_positions = jnp.array([
            (x, y) for x in range(1, h - 1) for y in range(1, w - 1)
        ])
        self.obs_size = 2 * self.view_range + 1
    
    @property
    def default_const(self) -> GridWorldConst:
        return GridWorldConst(max_steps=self.max_steps, goal_detect_reward=0.01)
    
    @property
    def env_observation_space(self) -> tuple[ObservationSpace, dict[str, tuple[int]]]:
        map_n_feature = 1
        my_pos_n_feature = 1
        other_pos_n_feature = self.n_agent - 1
        goal_pos_n_feature = 1
        total_n_feature = map_n_feature + my_pos_n_feature + goal_pos_n_feature + other_pos_n_feature
        obs_space = (self.obs_size, self.obs_size, total_n_feature)
        return ObservationSpace({'grid': obs_space}), {'grid': (self.n_agent,)}
    
    @property
    def env_action_space(self) -> tuple[ActionSpace, dict[str, tuple[int]]]:
        num_actions = len(Actions)
        return ActionSpace({'all': num_actions}), {'all': (self.n_agent,)}

    def _reset_positions(self, rng: jax.Array, const: GridWorldConst) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Reset the map, and the positions of all agents and goals
        """
        h, w = self.grid_size, self.grid_size
        n_agent, n_goal, n_clutter = self.n_agent, self.n_goal, self.n_clutter
        available_positions = self.available_positions

        rng, rng_index = jax.random.split(rng)
        indices = jax.random.choice(rng_index, len(available_positions), shape=(n_agent + n_goal + n_clutter,), replace=False)
        agent_indicies, goal_indicies, clutter_indicies = indices[:n_agent], indices[n_agent:n_agent+n_goal], indices[n_agent+n_goal:]
        
        agent_pos = available_positions[agent_indicies]
        goal_pos = available_positions[goal_indicies]
        clutter_pos = available_positions[clutter_indicies]
        return agent_pos, goal_pos, clutter_pos

    def env_reset(
        self,
        rng: jax.Array,
        const: GridWorldConst
    ) -> tuple[GridWorldState, Observation]:
        """Resets the environment to an initial state."""
        h, w = self.grid_size, self.grid_size

        rng, rng_reset_pos = jax.random.split(rng)
        agent_pos, goal_pos, clutter_pos = self._reset_positions(rng_reset_pos, const)
        
        rng, rng_score = jax.random.split(rng)
        goal_scores = self.goal_score_scale * jax.random.uniform(rng_score, shape=(self.n_goal,), minval=-1, maxval=1)
        goal_pos_map = jnp.zeros((h, w), dtype=jnp.bool_)
        goal_pos_map = goal_pos_map.at[goal_pos[:, 0], goal_pos[:, 1]].set(True)
        goal_score_map = jnp.zeros((h, w), dtype=jnp.float32)
        goal_score_map = goal_score_map.at[goal_pos[:, 0], goal_pos[:, 1]].set(goal_scores)

        wall_map = jnp.zeros((h, w), dtype=jnp.bool_)
        wall_map = wall_map.at[0, :].set(True)
        wall_map = wall_map.at[-1, :].set(True)
        wall_map = wall_map.at[:, 0].set(True)
        wall_map = wall_map.at[:, -1].set(True)
        wall_map = wall_map.at[clutter_pos[:, 0], clutter_pos[:, 1]].set(True)

        state = GridWorldState(
            _step=0,
            agent_pos=agent_pos,
            agent_allow_move=jnp.ones((self.n_agent,), dtype=jnp.bool_),
            agent_trial_best_score=-jnp.ones((self.n_agent,), dtype=jnp.float32),
            agent_discount_trial_ret=jnp.zeros((self.n_agent,), dtype=jnp.float32),
            goal_pos_map=goal_pos_map,
            goal_score_map=goal_score_map,
            wall_map=wall_map,
        )
        obs = self.get_obs(state)
        return jax.lax.stop_gradient(state), jax.lax.stop_gradient(obs)

    def _crop_map(self, full_map: jax.Array, pos: jax.Array) -> jax.Array:
        full_map = jnp.pad(full_map, self.view_range)
        return jax.lax.dynamic_slice(full_map, pos, (self.obs_size, self.obs_size))
    
    def get_agent_obs(self, state: GridWorldState, agent_id: int) -> jax.Array:
        agent_pos = state.agent_pos[agent_id]
        agent_trial_best_score = state.agent_trial_best_score[agent_id]
        h, w = self.grid_size, self.grid_size
        empty_map = jnp.zeros((h, w), dtype=jnp.float32)

        pos_map = empty_map.at[agent_pos[0], agent_pos[1]].set(2 + agent_trial_best_score)
        pos_map = self._crop_map(pos_map, agent_pos)
        goal_map = self._crop_map(state.goal_pos_map, agent_pos)
        wall_map = self._crop_map(state.wall_map, agent_pos)

        return jnp.concatenate([
            wall_map.reshape(self.obs_size, self.obs_size, 1),
            pos_map.reshape(self.obs_size, self.obs_size, 1),
            goal_map.reshape(self.obs_size, self.obs_size, 1),
        ], axis=-1)

    def get_obs(self, state: GridWorldState) -> Observation:
        my_obs = jax.vmap(self.get_agent_obs, in_axes=(None, 0))(state, jnp.arange(self.n_agent))

        h, w = self.grid_size, self.grid_size
        empty_map = jnp.zeros((h, w), dtype=jnp.float32)
        other_obs = jnp.stack([
            jnp.stack([
                self._crop_map(empty_map.at[state.agent_pos[j, 0], state.agent_pos[j, 1]].set(2 + state.agent_discount_trial_ret[j]), state.agent_pos[i])
                for j in range(self.n_agent) if i != j
            ], axis=-1)
            for i in range(self.n_agent)
        ], axis=0)

        return Observation({
            'grid': jnp.concatenate([my_obs, other_obs], axis=-1)
        })

    def agent_step(self, rng: jax.Array, const: GridWorldConst, state: GridWorldState, agent_id: int, action: int) -> dict[str, jax.Array]:
        agent_pos = state.agent_pos[agent_id]
        agent_allow_move = state.agent_allow_move[agent_id]
        agent_trial_best_score = state.agent_trial_best_score[agent_id]

        move = jnp.array([0, 0])
        move = jax.lax.cond(action == Actions.UP, lambda _: jnp.array([-1, 0]), lambda _: move, None)
        move = jax.lax.cond(action == Actions.DOWN, lambda _: jnp.array([1, 0]), lambda _: move, None)
        move = jax.lax.cond(action == Actions.LEFT, lambda _: jnp.array([0, -1]), lambda _: move, None)
        move = jax.lax.cond(action == Actions.RIGHT, lambda _: jnp.array([0, 1]), lambda _: move, None)

        move = jax.lax.cond(agent_allow_move, lambda _: move, lambda _: jnp.array([0, 0]), None)

        new_pos = agent_pos + move

        collide_wall = state.wall_map[new_pos[0], new_pos[1]]
        new_pos = jax.lax.cond(collide_wall, lambda _: agent_pos, lambda _: new_pos, None)

        reach_goal = state.goal_pos_map[new_pos[0], new_pos[1]]

        score = reach_goal * state.goal_score_map[new_pos[0], new_pos[1]] + (1 - reach_goal) * -1

        goal_pos_in_view = self._crop_map(state.goal_pos_map, agent_pos)
        goal_detected = jnp.any(goal_pos_in_view)

        reward = score - agent_trial_best_score
        reward += const.goal_detect_reward * goal_detected
        
        return {
            'new_pos': new_pos,
            'reach_goal': reach_goal,
            'score': score,
            'goal_detected': goal_detected,
            'best_score': jnp.maximum(score, agent_trial_best_score),
            'reward': reward
        }
    
    def _reset_for_trial(self, rng: jax.Array, const: GridWorldConst, state: GridWorldState) -> GridWorldState:
        rng, rng_reset_pos = jax.random.split(rng)
        agent_pos, goal_pos, clutter_pos = self._reset_positions(rng_reset_pos, const)
        return state.replace(
            agent_pos=agent_pos,
            agent_allow_move=jnp.ones((self.n_agent,), dtype=jnp.bool_),
            agent_trial_best_score=-jnp.ones((self.n_agent,), dtype=jnp.float32),
            agent_discount_trial_ret=self.last_ret_scale * state.agent_discount_trial_ret + self.history_ret_scale * state.agent_trial_best_score 
        )

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
            agent_pos=data['new_pos'],
            agent_trial_best_score=data['best_score'],
        )
        if self.stop_after_goal_reach:
            agent_allow_move = jnp.logical_and(
                state.agent_allow_move,
                jnp.logical_not(data['reach_goal'])
            )
            new_state = new_state.replace(agent_allow_move=agent_allow_move)        
        
        # Reset for each trial
        rng, rng_trial = jax.random.split(rng)
        state_new_trial = self._reset_for_trial(rng_trial, const, new_state)
        new_state = jax.lax.cond(num_steps % self.trial_steps == 0, lambda _: state_new_trial, lambda _: new_state, None)

        rng, rng_reset = jax.random.split(rng)
        state_reset, obs_reset = self.env_reset(rng_reset, const)

        return jax.lax.cond(
            done,
            lambda _: (state_reset, obs_reset, reward, jnp.repeat(done, self.n_agent), {}),
            lambda _: (new_state, self.get_obs(new_state), reward, jnp.repeat(done, self.n_agent), {}),
            None
        )
    
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
    
    def export_gif(self, states: list[GridWorldState], agent_id: int = 0, filename: str = "gridworld.gif"):
        """Creates a GIF from a list of states."""
        fig, ax = plt.subplots(figsize=(self.grid_size / 2, self.grid_size / 2))
    
        def update(frame: int) -> None:
            ax.clear()
            state = states[frame]
    
            grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
            grid[:] = " "
            grid[state.wall_map] = "#"

            bg_map = np.array(0.3 * (1 - state.wall_map))
            for i in range(self.n_agent):
                x, y = int(state.agent_pos[i][0].item()), int(state.agent_pos[i][1].item())
                bg_map[max(0, x - self.view_range):min(self.grid_size, x + self.view_range + 1), max(0, y - self.view_range):min(self.grid_size, y + self.view_range + 1)] = 1.0

            ax.imshow(bg_map, cmap="gray", alpha=0.3)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if grid[i, j] == " ":
                        continue
                    ax.text(j, i, grid[i, j], ha="center", va="center", fontsize=12)
            
            blue = jnp.array([0,0,1.0])
            red = jnp.array([1.0,0,0])


            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if not state.goal_pos_map[i, j]:
                        continue
                    s = jnp.clip(state.goal_score_map[i, j], -1, 1)
                    c = (red * jnp.clip(s, 0, 1) + blue * jnp.clip(-s, 0, 1)).tolist()
                    square = patches.Rectangle(
                        (j - 0.5, i - 0.5),  # lower-left corner
                        1, 1,                                                        # width and height
                        facecolor=c,
                        edgecolor=c
                    )
                    ax.add_patch(square)

            for i in range(self.n_agent): 
                s = jnp.clip(state.agent_discount_trial_ret[i], -1, 1)
                c = (red * jnp.clip(s, 0, 1) + blue * jnp.clip(-s, 0, 1)).tolist()
                square = patches.Rectangle(
                    (state.agent_pos[i][1] - 0.5, state.agent_pos[i][0] - 0.5),  # lower-left corner
                    1, 1,                                                        # width and height
                    facecolor=c,
                    edgecolor='black'
                )
                ax.add_patch(square)
                ax.text(state.agent_pos[i][1], state.agent_pos[i][0], f"{i}", ha="center", va="center", fontsize=14, color="white")

            ax.set_xticks([])
            ax.set_yticks([])

        ani = animation.FuncAnimation(fig, update, frames=len(states), interval=500)
        ani.save(filename, writer="imagemagick")
        plt.close(fig)

class NaiveFourPlayerGame(GridWorldEnv):
    def __init__(self, num_agents = 4, num_goals = 4, grid_size = 20, view_range = 5, clutter_density = 0, trial_steps = 20, max_steps = 120, goal_score_scale = 1, stop_after_goal_reach = True, last_ret_scale = 0.4, history_ret_scale = 0.6):
        assert num_agents == 4
        assert num_goals == 4
        assert int(clutter_density * (grid_size - 2) * (grid_size - 2)) == 0
        super().__init__(num_agents, num_goals, grid_size, view_range, clutter_density, trial_steps, max_steps, goal_score_scale, stop_after_goal_reach, last_ret_scale, history_ret_scale)
    
    def _reset_positions(self, rng: jax.Array, const: GridWorldConst) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Reset the map, and the positions of all agents and goals
        """
        h, w = self.grid_size, self.grid_size
        n_agent, n_goal, n_clutter = self.n_agent, self.n_goal, self.n_clutter
        available_positions = self.available_positions

        rng, rng_index = jax.random.split(rng)
        indices = jax.random.choice(rng_index, len(available_positions), shape=(n_agent + n_goal + n_clutter,), replace=False)
        agent_indicies, goal_indicies, clutter_indicies = indices[:n_agent], indices[n_agent:n_agent+n_goal], indices[n_agent+n_goal:]
        
        agent_pos = available_positions[agent_indicies]
        goal_pos = available_positions[goal_indicies]
        clutter_pos = available_positions[clutter_indicies]

        agent_area_size = self.view_range // 2 + 1
        goal_area_size = (self.grid_size - self.view_range) // 2

        agent_pos %= agent_area_size
        agent_pos += 1
        agent_pos = jnp.array([
            [h // 2 - agent_pos[0, 0], w // 2 - agent_pos[0, 1]],
            [h // 2 - agent_pos[1, 0], w // 2 + agent_pos[1, 1]],
            [h // 2 + agent_pos[2, 0], w // 2 - agent_pos[2, 1]],
            [h // 2 + agent_pos[3, 0], w // 2 + agent_pos[3, 1]],
        ])

        goal_pos %= goal_area_size
        goal_pos = jnp.array([
            [1 + goal_pos[0, 0], 1 + goal_pos[0, 1]],
            [1 + goal_pos[1, 0], w - 2 - goal_pos[1, 1]],
            [h - 2 - goal_pos[2, 0], 1 + goal_pos[2, 1]],
            [h - 2 - goal_pos[3, 0], w - 2 - goal_pos[3, 1]],
        ])
        return agent_pos, goal_pos, clutter_pos
    
    def env_reset(
        self,
        rng: jax.Array,
        const: GridWorldConst
    ) -> tuple[GridWorldState, Observation]:
        """Resets the environment to an initial state."""
        h, w = self.grid_size, self.grid_size

        rng, rng_reset_pos = jax.random.split(rng)
        agent_pos, goal_pos, clutter_pos = self._reset_positions(rng_reset_pos, const)
        
        rng, rng_score = jax.random.split(rng)
        goal_scores = 0.10 * self.goal_score_scale * jax.random.uniform(rng_score, shape=(self.n_goal,), minval=-1, maxval=1)
        rng, rng_base_perm = jax.random.split(rng)
        goal_scores += jax.random.permutation(rng_base_perm, jnp.array([-0.8, -0.2, 0.2, 0.8], dtype=jnp.float32))
        goal_pos_map = jnp.zeros((h, w), dtype=jnp.bool_)
        goal_pos_map = goal_pos_map.at[goal_pos[:, 0], goal_pos[:, 1]].set(True)
        goal_score_map = jnp.zeros((h, w), dtype=jnp.float32)
        goal_score_map = goal_score_map.at[goal_pos[:, 0], goal_pos[:, 1]].set(goal_scores)

        wall_map = jnp.zeros((h, w), dtype=jnp.bool_)
        wall_map = wall_map.at[0, :].set(True)
        wall_map = wall_map.at[-1, :].set(True)
        wall_map = wall_map.at[:, 0].set(True)
        wall_map = wall_map.at[:, -1].set(True)
        wall_map = wall_map.at[clutter_pos[:, 0], clutter_pos[:, 1]].set(True)

        state = GridWorldState(
            _step=0,
            agent_pos=agent_pos,
            agent_allow_move=jnp.ones((self.n_agent,), dtype=jnp.bool_),
            agent_trial_best_score=-jnp.ones((self.n_agent,), dtype=jnp.float32),
            agent_discount_trial_ret=jnp.zeros((self.n_agent,), dtype=jnp.float32),
            goal_pos_map=goal_pos_map,
            goal_score_map=goal_score_map,
            wall_map=wall_map,
        )
        obs = self.get_obs(state)
        return jax.lax.stop_gradient(state), jax.lax.stop_gradient(obs)

class TwoRoundFourPlayerGame(NaiveFourPlayerGame):
    def __init__(self, num_agents = 4, num_goals = 4, grid_size = 20, view_range = 5, clutter_density = 0, trial_steps = 20, max_steps = 40, goal_score_scale = 1, stop_after_goal_reach = True, last_ret_scale = 0.4, history_ret_scale = 0.6):
        assert num_agents == 4
        assert num_goals == 4
        assert int(clutter_density * (grid_size - 2) * (grid_size - 2)) == 0
        assert max_steps == 2 * trial_steps
        super().__init__(num_agents, num_goals, grid_size, view_range, clutter_density, trial_steps, max_steps, goal_score_scale, stop_after_goal_reach, last_ret_scale, history_ret_scale)
        
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

        # Additional rewards at the second trial
        second_trial_reward = jax.lax.cond(
            num_steps == 2 * self.trial_steps,
            lambda _: 10 * data['best_score'],
            lambda _: 0 * data['best_score'],
            None
        )
        agent_occupy_map = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.bool_).at[state.agent_pos[:, 0], state.agent_pos[:, 1]].set(True)
        num_goal_covered = (state.goal_pos_map * agent_occupy_map).astype(jnp.float32).sum()
        is_hit_best = second_trial_reward >= 10 * 0.5
        reward += second_trial_reward


        done = num_steps >= const.max_steps
        new_state = state.replace(
            _step=state._step + 1,
            agent_pos=data['new_pos'],
            agent_trial_best_score=data['best_score'],
        )
        if self.stop_after_goal_reach:
            agent_allow_move = jnp.logical_and(
                state.agent_allow_move,
                jnp.logical_not(data['reach_goal'])
            )
            new_state = new_state.replace(agent_allow_move=agent_allow_move)        
        
        # Reset for each trial
        rng, rng_trial = jax.random.split(rng)
        state_new_trial = self._reset_for_trial(rng_trial, const, new_state)
        new_state = jax.lax.cond(num_steps % self.trial_steps == 0, lambda _: state_new_trial, lambda _: new_state, None)

        rng, rng_reset = jax.random.split(rng)
        state_reset, obs_reset = self.env_reset(rng_reset, const)

        return jax.lax.cond(
            done,
            lambda _: (state_reset, obs_reset, reward, jnp.repeat(done, self.n_agent), {'num_goal_covered': jnp.zeros_like(num_goal_covered), 'is_hit_best': jnp.zeros_like(is_hit_best)}),
            lambda _: (new_state, self.get_obs(new_state), reward, jnp.repeat(done, self.n_agent), {'num_goal_covered': num_goal_covered, 'is_hit_best': is_hit_best}),
            None
        )

class TwoRoundSinglePlayerGame(TwoRoundFourPlayerGame):
    def get_obs(self, state: GridWorldState) -> Observation:
        my_obs = jax.vmap(self.get_agent_obs, in_axes=(None, 0))(state, jnp.arange(self.n_agent))

        h, w = self.grid_size, self.grid_size
        empty_map = jnp.zeros((h, w), dtype=jnp.float32)
        other_obs = jnp.stack([
            jnp.stack([
                self._crop_map(empty_map, state.agent_pos[i]) # Zero out the position of other agents
                for j in range(self.n_agent) if i != j
            ], axis=-1)
            for i in range(self.n_agent)
        ], axis=0)

        return Observation({
            'grid': jnp.concatenate([my_obs, other_obs], axis=-1)
        })