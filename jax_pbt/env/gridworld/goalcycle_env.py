from enum import IntEnum
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from ..base_env import BaseEnvConst, BaseEnvState, BaseEnv
from ..spaces import Observation, Action, ObservationSpace, ActionSpace

class GoalCycleConst(BaseEnvConst):
    goal_reward: float
    goal_penalty: float
    distance_penalty: float

class GoalCycleState(BaseEnvState):
    agent_pos: jax.Array
    agent_last_visited_goal: jax.Array
    goal_pos: jax.Array
    wall_map: jax.Array

class Actions(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2 
    RIGHT = 3 
    STAY = 4

class GoalCycleEnv(BaseEnv[GoalCycleConst, GoalCycleState]):
    def __init__(
        self, 
        num_agents: int = 1,
        num_goals: int = 3,
        grid_size: int = 13, 
        max_steps: int = 60, 
        distance_penalty: float = 0.0, 
        goal_reward: float = 1.0,
        goal_penalty: float = -1.0, 
        clutter_density: float = 0.1
    ) -> None:
        """
        [Param]
            num_agents: Number of players
            num_goals: NUmber of goals.
            grid_size: The size of the grid (grid_size x grid_size).
            max_steps: Maximum steps before the environment terminates.
            distance_penalty: Penalty for each step based on Manhattan distance to the goal.
            goal_reward: Reward for reaching the goal.
            goal_penalty: Penalty for incorrect goal order visiting.
            clutter_density: Ratio of number_of_clutters / grid_world_size for clutter generation
        """
        self.n_agent = num_agents
        self.n_goal = num_goals
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.distance_penalty = distance_penalty
        self.goal_reward = goal_reward
        self.goal_penalty = goal_penalty
        self.n_clutter = int(clutter_density * (grid_size - 2) * (grid_size - 2))

        h, w = self.grid_size, self.grid_size
        self.available_positions = jnp.array([
            (x, y) for x in range(1, h - 1) for y in range(1, w - 1)
        ])
        # Currently the goal order is 1...n by default; Customized orders need further development
    
    @property
    def default_const(self) -> GoalCycleConst:
        return GoalCycleConst(max_steps=self.max_steps, goal_reward=self.goal_reward, goal_penalty=self.goal_penalty, distance_penalty=self.distance_penalty)
    
    @property
    def env_observation_space(self) -> tuple[ObservationSpace, dict[str, tuple[int]]]:
        map_n_feature = 1
        pos_n_feature = 1
        goal_n_feature = self.n_goal
        total_n_feature = map_n_feature + pos_n_feature + goal_n_feature
        obs_space = (self.grid_size, self.grid_size, total_n_feature)
        return ObservationSpace({'grid': obs_space}), {'grid': (self.n_agent,)}
    
    @property
    def env_action_space(self) -> tuple[ActionSpace, dict[str, tuple[int]]]:
        num_actions = len(Actions)
        return ActionSpace({'all': num_actions}), {'all': (self.n_agent,)}
    
    def env_reset(
        self,
        rng: jax.Array,
        const: GoalCycleConst
    ) -> tuple[GoalCycleState, Observation]:
        """Resets the environment to an initial state."""
        h, w = self.grid_size, self.grid_size
        n_agent, n_goal, n_clutter = self.n_agent, self.n_goal, self.n_clutter
        
        # Initialize wall map with outer walls
        wall_map = jnp.zeros((h, w), dtype=jnp.bool_)
        wall_map = wall_map.at[0, :].set(True)
        wall_map = wall_map.at[-1, :].set(True)
        wall_map = wall_map.at[:, 0].set(True)
        wall_map = wall_map.at[:, -1].set(True)

        available_positions = self.available_positions
    
        rng, rng_index = jax.random.split(rng)
        indices =jax.random.choice(rng_index, len(available_positions), shape=(n_agent + n_goal + n_clutter,), replace=False)

        agent_indicies, goal_indicies, clutter_indicies = indices[:n_agent], indices[n_agent:n_agent+n_goal], indices[n_agent+n_goal:]
        agent_pos = available_positions[agent_indicies]
        goal_pos = available_positions[goal_indicies]
        clutter_pos = available_positions[clutter_indicies]
        wall_map = wall_map.at[clutter_pos[:, 0], clutter_pos[:, 1]].set(True)

        state = GoalCycleState(
            _step=0,
            agent_pos=agent_pos,
            agent_last_visited_goal=-jnp.ones((self.n_agent,), dtype=jnp.int32),
            goal_pos=goal_pos,
            wall_map=wall_map
        )
        obs = self.get_obs(state)
        return jax.lax.stop_gradient(state), jax.lax.stop_gradient(obs)
    
    def get_agent_obs(self, state: GoalCycleState, agent_id: int) -> Observation:
        agent_pos = state.agent_pos[agent_id]
        h, w = self.grid_size, self.grid_size
        empty_map = jnp.zeros((h, w), dtype=jnp.int8)
        pos_map = empty_map.at[agent_pos[0], agent_pos[1]].set(1)
        def set_goal_map(single_goal_pos) -> jax.Array:
            return empty_map.at[single_goal_pos[0], single_goal_pos[1]].set(1)
        goal_map = jax.vmap(set_goal_map, out_axes=-1)(state.goal_pos)
        return Observation({
            'grid': jnp.concatenate([
                state.wall_map.reshape(h, w, 1),
                pos_map.reshape(h, w, 1),
                goal_map
            ], axis=-1)
        })

    def get_obs(self, state: GoalCycleState) -> Observation:
        obs = jax.vmap(self.get_agent_obs, in_axes=(None, 0))(state, jnp.arange(self.n_agent))
        return obs

    def agent_step(self, rng: jax.Array, const: GoalCycleConst, state: GoalCycleState, agent_id: int, action: int) -> dict[str, jax.Array]:
        agent_pos = state.agent_pos[agent_id]
        agent_last_visited_goal = state.agent_last_visited_goal[agent_id]

        move = jnp.array([0, 0])
        move = jax.lax.cond(action == Actions.UP, lambda _: jnp.array([-1, 0]), lambda _: move, None)
        move = jax.lax.cond(action == Actions.DOWN, lambda _: jnp.array([1, 0]), lambda _: move, None)
        move = jax.lax.cond(action == Actions.LEFT, lambda _: jnp.array([0, -1]), lambda _: move, None)
        move = jax.lax.cond(action == Actions.RIGHT, lambda _: jnp.array([0, 1]), lambda _: move, None)

        new_pos = agent_pos + move
        collides = state.wall_map[new_pos[0], new_pos[1]]
        new_pos = jax.lax.cond(collides, lambda _: agent_pos, lambda _: new_pos, None)

        is_moving = jnp.any(agent_pos != new_pos)
        is_goal_hit = jax.vmap(lambda goal_pos: jnp.logical_and(jnp.all(goal_pos == new_pos), is_moving))(state.goal_pos)
        goal_id = jnp.select(is_goal_hit, jnp.arange(self.n_goal), -1)
        is_correct_goal_hit = jnp.logical_and(goal_id >= 0, jnp.logical_or(agent_last_visited_goal == -1, goal_id == (agent_last_visited_goal + 1) % self.n_goal))
        is_incorrect_goal_hit = jnp.logical_and(goal_id >= 0, jnp.logical_and(agent_last_visited_goal != -1, goal_id != (agent_last_visited_goal + 1) % self.n_goal))

        new_agent_last_visited_goal = jax.lax.cond(is_correct_goal_hit, lambda _: goal_id, lambda _: agent_last_visited_goal, None)
        reward = const.goal_reward * is_correct_goal_hit + const.goal_penalty * is_incorrect_goal_hit
        return {
            'new_pos': new_pos,
            'new_agent_last_visited_goal': new_agent_last_visited_goal,
            'reward': reward
        }

    def env_step(
        self,
        rng: jax.Array,
        const: GoalCycleConst,
        state: GoalCycleState,
        action: Action
    ) -> tuple[GoalCycleState, Observation, jax.Array, jax.Array, dict[Any, Any]]:
        env_action = action['all']
        rng, rng_agent_step_batch = jax.random.split(rng)
        rng_agent_step_batch = jax.random.split(rng_agent_step_batch, self.n_agent)
        data = jax.vmap(self.agent_step, in_axes=(0, None, None, 0, 0))(rng_agent_step_batch, const, state, jnp.arange(self.n_agent), env_action)
        
        num_steps = state._step + 1
        done = num_steps >= const.max_steps
        new_state = state.replace(
            _step=state._step + 1,
            agent_pos=data['new_pos'],
            agent_last_visited_goal=data['new_agent_last_visited_goal']
        )
        reward = data['reward']

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
        const: GoalCycleConst
    ) -> tuple[GoalCycleState, list[Observation]]:
        state, env_obs = self.env_reset(rng, const)
        obs_lst = [env_obs[i] for i in range(self.n_agent)]
        return state, obs_lst
    
    def step(
        self,
        rng: jax.Array,
        const: GoalCycleConst,
        state: GoalCycleState,
        action: Sequence[Action]
    ) -> tuple[GoalCycleState, list[Observation], list[jax.Array], list[jax.Array], dict[Any, Any]]:
        env_action = Action({
            'all': jnp.stack([agent_action['all'] for agent_action in action])
        })
        state, env_obs, env_reward, env_done, info = self.env_step(rng, const, state, env_action)
        obs_lst = [env_obs[i] for i in range(self.n_agent)]
        reward_lst = [env_reward[i] for i in range(self.n_agent)]
        done_lst = [env_done[i] for i in range(self.n_agent)]
        return state, obs_lst, reward_lst, done_lst, info
    
    def visualize_states(self, states: list[GoalCycleState], rewards: list[float], agent_id: int = 0, filename: str = "gridworld.gif"):
        """Creates a GIF from a list of states."""
        fig, ax = plt.subplots(figsize=(self.grid_size / 2, self.grid_size / 2))

        cum_rewards = jnp.cumsum(np.array(rewards))

        def update(frame: int) -> None:
            ax.clear()
            state = states[frame]
            grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
            grid[:] = " "
            grid[state.wall_map] = "#"
            for i in range(self.n_goal):
                grid[state.goal_pos[i][0], state.goal_pos[i][1]] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdedfhijklmnopqrstuvwxyz"[i]
            for i in range(self.n_agent):
                grid[state.agent_pos[i][0], state.agent_pos[i][1]] = f"{i}"
            ax.imshow(1 - state.wall_map, cmap="gray", alpha=0.3)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    ax.text(j, i, grid[i, j], ha="center", va="center", fontsize=12)

            reward_text = f"collective reward: {cum_rewards[frame]} / {sum(rewards)}  last goal of agent {agent_id} :{'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdedfhijklmnopqrstuvwxyz'[state.agent_last_visited_goal[agent_id]]}  step:{frame}/{len(rewards)}"  # Assuming state.reward stores the reward at that step
            ax.text(0.5, -0.1, reward_text, ha="center", va="center", fontsize=14, transform=ax.transAxes)

            ax.set_xticks([])
            ax.set_yticks([])

        ani = animation.FuncAnimation(fig, update, frames=len(states), interval=500)
        ani.save(filename, writer="imagemagick")
        plt.close(fig)