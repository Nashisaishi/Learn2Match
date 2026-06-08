# Environment


The environment can be used with normal RL environment APIs, as shown below:

```python
import jax
from jax_pbt.env.gridworld.collect_env import GridWorldEnv as Env

rng = jax.random.key(0)
# env initialization with the default constant
env_fn = Env(num_agents=2, max_steps=60)
env_const = env_fn.default_const

env_state, obs_lst = env_fn.reset(rng, env_const) # obs_lst contains observation for each (RL-controlled) agent
env_state_seq = [env_state] # save env_state for rendering
for t in range(60):
    rng, rng_action_lst = jax.random.split(rng)
    rng_action_lst = list(jax.random.split(rng_action_lst))
    action_lst = [action_space.sample(rng_action) for rng_action, action_space in zip(rng_action_lst, env_fn.get_action_space())]
    env_state, obs_lst, reward_lst, done_lst, info = env_fn.step(rng, env_const, env_state, action_lst)
    env_state_seq.append(env_state)

env_fn.export_gif(env_state_seq, filename="readme_example_trajectory.gif")
```

There are a few key concepts for environments:

- **EnvConst**: constants or parameters for environment ***reset*** or ***step***.
    - It must have a *max_steps* attribute, indicating the maximum length of an episode.
- **EnvState**: the state of the environment on the current state, initialized by ***reset*** and will be updated by ***step***.
    - It must have a *_step* attribute, indicating the timesteps pasted for the state.
- Environment externel APIs, observation, action: will be discussed soon. Can be found in [](base_env.py) and [](spaces.py).



## Quick view on key functions

- *default_const*
    - [Property] return the default **EnvConst**
- Internal
    - *env_observation_space/env_action_space*: environment observation/action space, and the number (shape) of environment agents that receives each environment atttibute
    - *env_reset/env_step*: environment reset/step
- *num_agents*: number of RL agents/policies to provide actions to the environment, exposed as an external API
    - [Property] number of agents
- *get_observation_space()/get_action_space()*: a list observation/action space for each RL agent, exposed
    - [Output] list[ObservationSpace] or list[ActionSpace]
- *get_agent_batch_shape()*: if one agent has batch_shape and observation_shape, then the agent would receive (*batch_shape, *observation_shape) 
    - [Output] a list of shapes (list[tuple[int]])
- ***reset()/step()***: external reset/step, exposed
    - For ***reset**
        - [Input] rng, **EnvConst**
        - [Output] **EnvState**, list[Observation] (The list of the observation should be the same as env.num_agents)
    - For ***step**
        - [Input] rng, **EnvConst**, **EnvState**, list[Action]
        - [Output] **EnvState**, list[Observation], reward list, done flag list, information dict



## Design choice

We make a separation of interval APIs/functions and external APIs/functions. The previous ones only focuse on the dynamics of the environment (i.e., as a core simulator/dynamic), and the later one expose APIs to externel RL agents.



## Observation and action (space)

The observation and action are essentially `dict[str, jax.Array]`. We provide `Observation` and `Action` as wrappers and you can generally use them like `jax.Array` (e.g., `Action.concatenate` $\approx$ `jax.numpy.concatenate`). # TODO: check all coding in all doc and and `code`

We will illustrate the design logic of the observation space and action space with an example.

Consider an environment with 3 humanoid robot with camera input (RGB image) and a vector of current position (3d vector), and 6 quadruped robot with position vector only as the input. The humanoid is controlled by high-level actions (with 7 different actions), and quadruped is controlled by joint input with a dgree-of-freedom (DOF) of 10.

Assume all humanoids are controlled by one RL agent/policy, and all quadrupeds are controlled by the other RL agent/policy. In that case, for a single environment, you might expect to get the following results for calling observation/action space query functions:

```Python
# Please use Observation, Action, ObservationSpace, ActionSpace. Although they are functionally the same as a dictionary, they provide additional attributes and methods required by other parts of the codebase

import jax
import jax.numpy as jnp

# for env_observation_space/env_action_space, it retruen a tuple, where the first represents the observation/action space for each attribute, and the second represents the number/shape of all agents that receives that atrtibute
env.env_observation_space = (
    ObservationSpace({
        'RGB': (84, 84, 3),
        'position': (3,)
    }),
    {
        'RGB': (3,) # only for 3 humanoid
        'position': (9,) # 3 humanoiud and 6 quadriped robots all receive this observation
    }
)

env.env_action_space = {
    ActionSpace({
        'humanoid-command': 7, # categorical; incdicating 7 different actions
        'quaruped-joint': (10,) # continuous with 10 dimensionts
    }),
    {
        'humanoid-command': (3,),
        'quaruped-joint': (6,)
    }
}

# The above spaces defines the shape of (internal) environment observation and environment action, which are expected to have the following shaped
next_state, env_obs, env_reward, env_done, info = env.env_step(rng, env_const, env_state, env_action)
env_obs = Observation({
    'RGB': jnp.zeros((3, 84, 84, 3), dtype=jnp.float32),
    'position': jnp.zeros((9, 3), dtype=jnp.float32)
})
env_action = Action({
    'humanoid-command': jnp.zeros((6,), dtype=jnp.int32) # from 0 ~ 6
    'quaruped-joint': jnp.zeros((6, 10), dtype=jnp.float32)
})
# Example for environmental agent rewards. Here we assume there are 9 (3 humanoid and 6 quariped) envronmental agents.
# But with the dict-style observation/action, the env_obs and env_action can be controlled more flexibly
env_reward = jnp.zeros((9,), dtype=jnp.float32)
env_done = jnp.zeros((9,), dtype=jnp.bool)


# ============= Now we have public APIs exposed to RL agents/policies ============


env.num_agents = 2 # two different RL agents/policies to control the robots

env.get_observation_space() = [
    ObservaionSpace({
        'RGB': (84, 84, 3),
        'position': (3,)
    }), # humanoids receive both features
    ObservaionSpace({
        'position': (3,)
    })
]

env.get_action_space() = [
    ActionSpace({
        'humanoid-command': 7,
    }),
    ActionSpace({
        'quaruped-joint': (3,)
    })
]

env.get_agent_batch_shape() = [(3,), (6,)]
```

In this case, you might expect the following observation and action (list) will be used for the ***step*** method of the environment.

```Python
next_env_state, obs_lst, reward_lst, done_lst, info = env.step(rng, env_const, env_state, action_lst)

obs_lst = [
    Observation({
        'RGB': jnp.zeros((3, 84, 84, 3), dtype=jnp.float32),
        'position': jnp.zeros((3, 3), dtype=jnp.float32)
    }),
    Observation({
        'position': jnp.zeros((6, 3), dtype=jnp.float32)
    })
]
action_lst = [
    Action({
        'humanoid-command': jnp.zeros((6,), dtype=jnp.int32) # from 0 ~ 6
    }),
    Action({
        'quaruped-joint': jnp.zeros((6, 10), dtype=jnp.float32)
    })
]
reward_lst = [jnp.zeros((3,), dtype=jnp.float32), jnp.zeros((6,), jnp.float32)]
done_lst = [jnp.zeros((3,), dtype=jnp.bool), jnp.zeros((6,), jnp.bool)]
```

## Environment dynamic

It is expected that the core dynamic is implemented in the *env_reset/env_step* function and only use ***reset/step*** to interact with externel RL agents/policies. Following the example environment above, you might expect an implementation of the ***step*** method as below:

```Python
def step(self, rng, const, state, action):
    env_action = Action({
        'humanoid-command': action[0]['humanoid-command'],
        'quaruped-joint': action[1]['quaruped-joint'],
    })
    state, env_obs, env_reward, env_done, info = self.env_step(rng, const, state, env_action)
    obs_lst = [
        Observation({
            'RGB': env_obs['RGB'],
            'position': env_obs['position'][:3]
        }),
        Observation({
            'position': env_obs['position'][3:]
        })
    ]
    reward_lst = [env_reward[:3], env_reward[3:]]
    done_lst = [env_done[:3], env_done[3:]]
    return state, obs_lst, reward_lst, done_lst, info
```

Please note that in Jax, we are no longer able to use control episode reset conditionally outside the environment; instead, a reset cretiria is checked in *env_step* for automatic episode reset.



## Batched environments

We provide a few environment stacking tools in [](batched_env.py) for parallel environment execution.



### BatchedEnv

Stacking environments with its original observation space, action space, and RL agent assignments.

```Python
batched_env = BatchedEnv(env, num_envs=100)
# batched_env.num_agents = env.num_agents
# batched_env.get_agent_batch_shape=[(100,)]
```



### HomogeneousBatchedEnv

If all agents in the environment are homogeneous and controlled by the same RL policy, all agents can be stacked togeter.

```Python
batched_env = HomogeneousBatchedEnv(env, num_envs=100)
# batched_env.num_agents = 1
# batched_env.get_agent_batch_shape=[(100 * env.num_agents,)]
```



### TODO: role assignment environment (and improve the doc string before, and use `code` for code)