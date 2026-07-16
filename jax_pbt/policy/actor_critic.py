from abc import ABC, abstractmethod
from typing import Sequence

import flax.linen as nn
import flax.struct as struct
from flax.struct import PyTreeNode
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

from ..env import Observation, Action, ObservationSpace, ActionSpace
from ..model.action_distribution import ActionDistribution, get_distribution_info, build_action_distribution_layer
from ..model.feature_extractors import build_feature_extractor_with_consistent_layers
from ..model.nn_blocks.mlp import MLP
from ..trainer.bc import BCTrainer
from ..trainer.ppo import PPOTrainer, PPOTransition
from ..utils import global_norm
from .agent_wrapper import RecurrentAgent, RecurrentPPOAgent


class BaseActor(nn.Module):
    obs_space: ObservationSpace = struct.field(pytree_node=False)
    action_space: ActionSpace = struct.field(pytree_node=False)
    def setup(self) -> None:
        self.feature_extractor = build_feature_extractor_with_consistent_layers(self.obs_space)
        self.policy_layers = build_action_distribution_layer(self.action_space)
    
    def __call__(self, rnn_states: jax.Array, obs: Observation) -> tuple[jax.Array, ActionDistribution]:
        rnn_states, feature, _ = self.feature_extractor(rnn_states, obs)
        pi = ActionDistribution({
            action_name: policy_layer(feature)
            for action_name, policy_layer in self.policy_layers.items()
        })
        return rnn_states, pi
    
    def init_rnn_state(self, dummy_batch: jax.Array) -> jax.Array:
        if self.feature_extractor.use_rnn:
            return jnp.zeros((*dummy_batch.shape, self.feature_extractor.rnn_hidden_layers * self.feature_extractor.rnn_hidden_size))
        else:
            return jnp.zeros(dummy_batch.shape)

class BaseCritic(nn.Module):
    obs_space: ObservationSpace = struct.field(pytree_node=False)
    def setup(self) -> None:
        self.feature_extractor = build_feature_extractor_with_consistent_layers(self.obs_space)
        self.value_layer = nn.Dense(1)
    
    def __call__(self, rnn_states: jax.Array, obs: Observation) -> tuple[jax.Array, jax.Array]:
        rnn_states, feature, _ = self.feature_extractor(rnn_states, obs)
        val = self.value_layer(feature)
        return rnn_states, val.squeeze(-1)
    
    def init_rnn_state(self, dummy_batch: jax.Array) -> jax.Array:
        if self.feature_extractor.use_rnn:
            return jnp.zeros((*dummy_batch.shape, self.feature_extractor.rnn_hidden_layers * self.feature_extractor.rnn_hidden_size))
        else:
            return jnp.zeros(dummy_batch.shape)

class BaseSharedActorCritic(nn.Module):
    obs_space: ObservationSpace = struct.field(pytree_node=False)
    action_space: ActionSpace = struct.field(pytree_node=False)
    def setup(self) -> None:
        self.feature_extractor = build_feature_extractor_with_consistent_layers(self.obs_space)
        self.actor_feature_extractor = lambda x: x
        self.critic_feature_extractor = lambda x: x
        self.policy_layers = build_action_distribution_layer(self.action_space)
        self.value_layer = nn.Dense(1)
    
    def __call__(self, rnn_states: jax.Array, obs: Observation) -> tuple[jax.Array, ActionDistribution, jax.Array]:
        rnn_states, feature, _ = self.feature_extractor(rnn_states, obs)
        actor_feature = self.actor_feature_extractor(feature)
        pi = ActionDistribution({
            action_name: policy_layer(actor_feature)
            for action_name, policy_layer in self.policy_layers.items()
        })
        critic_feature = self.critic_feature_extractor(feature)
        val = self.value_layer(critic_feature)
        return rnn_states, pi, val.squeeze(-1)
    
    def init_rnn_state(self, dummy_batch: jax.Array) -> jax.Array:
        if self.feature_extractor.use_rnn:
            return jnp.zeros((*dummy_batch.shape, self.feature_extractor.rnn_hidden_layers * self.feature_extractor.rnn_hidden_size))
        else:
            return jnp.zeros(dummy_batch.shape)

class SimpleActor(BaseActor):
    common_hidden_size: int = 64
    cnn_kernel_size: int = 3
    cnn_num_conv_layers: int = 3
    use_rnn: bool = False
    def setup(self) -> None:
        self.feature_extractor = build_feature_extractor_with_consistent_layers(
            self.obs_space,
            hidden_size=self.common_hidden_size,
            cnn_feature_lst=[self.common_hidden_size for _ in range(self.cnn_num_conv_layers)],
            cnn_kernel_lst=[(self.cnn_kernel_size, self.cnn_kernel_size) for _ in range(self.cnn_num_conv_layers)],
            use_rnn=self.use_rnn,
            rnn_hidden_layers=1,
            rnn_hidden_size=self.common_hidden_size
        )
        self.policy_layers = build_action_distribution_layer(self.action_space)

class SimpleCritic(BaseCritic):
    common_hidden_size: int = 64
    cnn_kernel_size: int = 3
    cnn_num_conv_layers: int = 3
    use_rnn: bool = False
    def setup(self) -> None:
        self.feature_extractor = build_feature_extractor_with_consistent_layers(
            self.obs_space,
            hidden_size=self.common_hidden_size,
            cnn_feature_lst=[self.common_hidden_size for _ in range(self.cnn_num_conv_layers)],
            cnn_kernel_lst=[(self.cnn_kernel_size, self.cnn_kernel_size) for _ in range(self.cnn_num_conv_layers)],
            use_rnn=self.use_rnn,
            rnn_hidden_layers=1,
            rnn_hidden_size=self.common_hidden_size
        )
        self.value_layer = nn.Dense(1)

class SimpleSharedActorCritic(BaseSharedActorCritic):
    common_hidden_size: int = 64
    cnn_kernel_size: int = 3
    cnn_num_conv_layers: int = 3
    use_rnn: bool = False
    def setup(self) -> None:
        self.feature_extractor = build_feature_extractor_with_consistent_layers(
            self.obs_space,
            hidden_size=self.common_hidden_size,
            cnn_feature_lst=[self.common_hidden_size for _ in range(self.cnn_num_conv_layers)],
            cnn_kernel_lst=[(self.cnn_kernel_size, self.cnn_kernel_size) for _ in range(self.cnn_num_conv_layers)],
            use_rnn=self.use_rnn,
            rnn_hidden_layers=1,
            rnn_hidden_size=self.common_hidden_size
        )
        self.actor_feature_extractor = lambda x: x
        self.critic_feature_extractor = lambda x: x
        self.policy_layers = build_action_distribution_layer(self.action_space)
        self.value_layer = nn.Dense(1)

class BaseActorCriticModel(ABC):
    @abstractmethod
    def init_model_state(self, rng: jax.Array) -> PyTreeNode:
        ...
    
    @abstractmethod
    def init_rnn_state(self, batch_shape: Sequence[int] = ()) -> PyTreeNode:
        ...
    
    def init_actor_rnn_state(self, batch_shape: Sequence[int] = ()) -> PyTreeNode:
        return self.init_rnn_state(batch_shape)

    @abstractmethod
    def forward(self, rng: jax.Array, model_state: PyTreeNode, rnn_state: PyTreeNode, obs: Observation) -> tuple[PyTreeNode, ActionDistribution, jax.Array]:
        ...

    @abstractmethod
    def get_action_distribution(self, rng: jax.Array, model_state: PyTreeNode, rnn_state: PyTreeNode, obs: Observation) -> tuple[PyTreeNode, ActionDistribution]:
        ...

    @abstractmethod
    def get_value(self, rng: jax.Array, model_state: PyTreeNode, rnn_state: PyTreeNode, obs: Observation) -> tuple[PyTreeNode, jax.Array]:
        ...
    
    @classmethod
    def output_nn(self, fn: BaseSharedActorCritic, obs_space: ObservationSpace, depth: int | None = -1) -> None:
        if depth != -1:
            print(fn.tabulate(jax.random.key(0), fn.apply({}, jnp.zeros(shape=()), method='init_rnn_state'), obs_space.example(batch_shape=(1,)), depth=depth))

class SeparatedActorCriticModel(BaseActorCriticModel):
    @classmethod
    def build_base_actor_critic(cls, obs_space: ObservationSpace, action_space: ActionSpace, output_nn_depth: int | None = -1) -> tuple[BaseActor, BaseCritic]:
        actor_fn = BaseActor(
            obs_space=obs_space,
            action_space=action_space,
        )
        critic_fn = BaseCritic(
            obs_space=obs_space,
        )
        cls.output_nn(actor_fn, obs_space, output_nn_depth)
        cls.output_nn(critic_fn, obs_space, output_nn_depth)
        return actor_fn, critic_fn
    
    @classmethod
    def build_simple_actor_critic(cls, obs_space: ObservationSpace, action_space: ActionSpace, common_hidden_size: int = 64, cnn_kernel_size: int = 3, conv_layers: int = 3, use_rnn: bool = False, output_nn_depth: int | None = -1) -> tuple[SimpleActor, SimpleCritic]:
        actor_fn = SimpleActor(
            obs_space=obs_space,
            action_space=action_space,
            common_hidden_size=common_hidden_size,
            cnn_kernel_size=cnn_kernel_size,
            cnn_num_conv_layers=conv_layers,
            use_rnn=use_rnn
        )
        critic_fn = SimpleCritic(
            obs_space=obs_space,
            common_hidden_size=common_hidden_size,
            cnn_kernel_size=cnn_kernel_size,
            use_rnn=use_rnn
        )
        cls.output_nn(actor_fn, obs_space, output_nn_depth)
        cls.output_nn(critic_fn, obs_space, output_nn_depth)
        return actor_fn, critic_fn
    
    def __init__(self, actor_fn: BaseActor, critic_fn: BaseCritic) -> None:
        self.actor_fn = actor_fn
        self.critic_fn = critic_fn

    def init_rnn_state(self, batch_shape: Sequence[int] = ()) -> PyTreeNode:
        actor_rnn_state = self.actor_fn.apply({}, jnp.zeros(batch_shape), method='init_rnn_state')
        critic_rnn_state = self.critic_fn.apply({}, jnp.zeros(batch_shape), method='init_rnn_state')
        return {'actor_rnn_state': actor_rnn_state, 'critic_rnn_state': critic_rnn_state}
    
    def init_actor_rnn_state(self, batch_shape: Sequence[int] = ()) -> PyTreeNode:
        actor_rnn_state = self.actor_fn.apply({}, jnp.zeros(batch_shape), method='init_rnn_state')
        return {'actor_rnn_state': actor_rnn_state}
    
    def init_model_state(self, rng: jax.Array) -> PyTreeNode:
        # An (unbatched) example with length 1 for obs
        rnn_state_example = self.init_rnn_state(batch_shape=())
        actor_obs_example = self.actor_fn.obs_space.example(batch_shape=(1,))
        critic_example = self.critic_fn.obs_space.example(batch_shape=(1,))

        actor_rng, critic_rng = jax.random.split(rng)
        actor_params = self.actor_fn.init(
            actor_rng,
            rnn_state_example['actor_rnn_state'],
            actor_obs_example
        )
        critic_params = self.critic_fn.init(
            critic_rng,
            rnn_state_example['critic_rnn_state'],
            critic_example
        )
        return {'actor_params': actor_params, 'critic_params': critic_params}
    
    def get_action_distribution(self, rng: jax.Array, model_state: PyTreeNode, rnn_state: PyTreeNode, obs: Observation) -> tuple[PyTreeNode, ActionDistribution]:
        """
        Computes the action distribution given the current model parameters, recurrent state, and observation.

        [Input]
            - rng: A random number generator used to control stochastic behavior in the actor model (e.g., dropout).
            - model_state:  A dictionary containing 'actor_params', which are the model parameters for the actor network.
            - rnn_state: A dictionary containing 'actor_rnn_state' with shape [*batch_shape, *rnn_state_shape],
                representing the hidden state of the recurrent network in the actor model.
            - obs: Input observation with shape [length, *batch_shape, *observation_shape]

        [Output]
            - next_rnn_state: A dictionary replacing 'actor_rnn_state' with next_actor_rnn_state,
                which represents the updated hidden state after processing the input observation.
            - action_distribution: A batch of action distributions computed from the actor network, with shape [length, *batch_shape].
        """

        actor_rnn_state = rnn_state['actor_rnn_state']
        actor_params = model_state['actor_params']
        next_actor_rnn_state, pi = self.actor_fn.apply(actor_params, actor_rnn_state, obs)
        return {'actor_rnn_state': next_actor_rnn_state}, pi
    
    def get_value(self, rng: jax.Array, model_state: PyTreeNode, rnn_state: PyTreeNode, obs: Observation) -> tuple[PyTreeNode, jax.Array]:
        """
        Computes the state value given the current model parameters, recurrent state, and observation.

        [Input]
            - rng: A random number generator used to control stochastic behavior in the critic model (e.g., dropout).
            - model_state:  A dictionary containing 'critic_params', which are the model parameters for the critic network.
            - rnn_state: A dictionary containing 'critic_rnn_state' with shape [*batch_shape, *rnn_state_shape],
                representing the hidden state of the recurrent network in the actor model.
            - obs: Input observation with shape [length, *batch_shape, *observation_shape]

        [Output]
            - next_rnn_state: A dictionary replacing 'critic_rnn_state' with next_critic_rnn_state,
                which represents the updated hidden state after processing the input observation.
            - value: A batch of values computed from the critic network, with shape [length, *batch_shape].
        """

        critic_rnn_state = rnn_state['critic_rnn_state']
        critic_params = model_state['critic_params']
        next_critic_rnn_state, val = self.critic_fn.apply(critic_params, critic_rnn_state, obs)
        return {'critic_rnn_state': next_critic_rnn_state}, val

    def forward(self, rng: jax.Array, model_state: PyTreeNode, rnn_state: PyTreeNode, obs: Observation) -> tuple[PyTreeNode, ActionDistribution, jax.Array]:
        actor_rng, critic_rng = jax.random.split(rng)
        next_actor_rnn_state, pi = self.get_action_distribution(actor_rng, model_state, rnn_state, obs)
        next_critic_rnn_state, val = self.get_value(critic_rng, model_state, rnn_state, obs)
        return {
            'actor_rnn_state': next_actor_rnn_state['actor_rnn_state'],
            'critic_rnn_state': next_critic_rnn_state['critic_rnn_state']
        }, pi, val

class SharedActorCriticModel(BaseActorCriticModel):
    @classmethod
    def build_base_shared_actor_critic(cls, obs_space: ObservationSpace, action_space: ActionSpace, output_nn_depth: int | None = -1) -> BaseSharedActorCritic:
        actor_critic_fn = BaseSharedActorCritic(
            obs_space=obs_space,
            action_space=action_space,
        )
        cls.output_nn(actor_critic_fn, obs_space, output_nn_depth)
        return actor_critic_fn
    
    @classmethod
    def build_simple_shared_actor_critic(cls, obs_space: ObservationSpace, action_space: ActionSpace, common_hidden_size: int = 64, cnn_kernel_size: int = 3, conv_layers: int = 3, use_rnn: bool = False, output_nn_depth: int | None = -1) -> SimpleSharedActorCritic:
        actor_critic_fn = SimpleSharedActorCritic(
            obs_space=obs_space,
            action_space=action_space,
            common_hidden_size=common_hidden_size,
            cnn_kernel_size=cnn_kernel_size,
            cnn_num_conv_layers=conv_layers,
            use_rnn=use_rnn
        )
        cls.output_nn(actor_critic_fn, obs_space, output_nn_depth)
        return actor_critic_fn
    
    def __init__(self, actor_critic_fn: BaseSharedActorCritic) -> None:
        self.actor_critic_fn = actor_critic_fn

    def init_rnn_state(self, batch_shape: Sequence[int] = ()) -> PyTreeNode:
        rnn_state = self.actor_critic_fn.apply({}, jnp.zeros(batch_shape), method='init_rnn_state')
        return {'rnn_state': rnn_state}
    
    def init_model_state(self, rng: jax.Array) -> PyTreeNode:
        # An (unbatched) example with length 1 for obs
        rnn_state_example = self.init_rnn_state(batch_shape=())
        obs_example = self.actor_critic_fn.obs_space.example(batch_shape=(1,))

        params = self.actor_critic_fn.init(
            rng,
            rnn_state_example['rnn_state'],
            obs_example
        )
        return {'actor_critic_params': params}
    
    def forward(self, rng: jax.Array, model_state: PyTreeNode, rnn_state: PyTreeNode, obs: Observation) -> tuple[PyTreeNode, ActionDistribution, jax.Array]:
        """
        Computes the action distribution and state values given the current model parameters, recurrent state, and observation.

        [Input]
            - rng: A random number generator used to control stochastic behavior in the actor-critic model (e.g., dropout).
            - model_state: A dictionary containing 'actor_critic_params', which are the parameters for the joint actor-critic network.
            - rnn_state:  A dictionary containing 'rnn_state' with shape with shape [*batch_shape, *rnn_state_shape], representing the hidden state of the recurrent network for the actor-critic model.
            - obs: Input observation with shape [length, *batch_shape, *observation_shape].

        [Output]
            - next_rnn_state: A dictionary replacing 'rnn_state' with next_rnn_state, representing the updated hidden state (RNN state) after processing the input observation.
            - action_distribution (pi): A batch of action distributions computed from the actor layer, with shape [length, *batch_shape].
            - value: A batch of values computed from the critic layer, with shape [length, *batch_shape].
        """
        next_rnn_state, pi, val = self.actor_critic_fn.apply(model_state['actor_critic_params'], rnn_state['rnn_state'], obs)
        return {'rnn_state': next_rnn_state}, pi, val
    
    def get_action_distribution(self, rng: jax.Array, model_state: PyTreeNode, rnn_state: PyTreeNode, obs: Observation) -> tuple[PyTreeNode, ActionDistribution]:
        next_rnn_state, pi, val = self.forward(rng, model_state, rnn_state, obs)
        return next_rnn_state, pi
    
    def get_value(self, rng: jax.Array, model_state: PyTreeNode, rnn_state: PyTreeNode, obs: Observation) -> tuple[PyTreeNode, jax.Array]:
        next_rnn_state, pi, val = self.forward(rng, model_state, rnn_state, obs)
        return next_rnn_state, val

class ActorWrapperAgent(RecurrentAgent):
    def __init__(self, actor_critic_fn: BaseActorCriticModel) -> None:
        self.actor_critic_fn = actor_critic_fn
    
    def init_agent_state(self, batch_shape: Sequence[int] = ()) -> PyTreeNode:
        return self.actor_critic_fn.init_actor_rnn_state(batch_shape)
    
    def reset_agent_state(self, agent_state: PyTreeNode, done: jax.Array) -> PyTreeNode:
        default_state = self.init_agent_state(done.shape)
        return jax.tree_util.tree_map(
            lambda x, y: x * (1 - done.reshape(done.shape + (1,) * (x.ndim - done.ndim))) + y, agent_state, default_state
        )
    
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
            - rng (jax.Array): Random number generator state for stochastic policies.
            - model_state (PyTreeNode): The agent's current model state.
            - agent_state (PyTreeNode): The agent's recurrent state (e.g., RNN hidden states).
            - obs (Observation): The observation received from the environment.

        [Output]
            - next_agent_state (PyTreeNode): Updated agent state (e.g., next RNN hidden state).
            - action (Action): The action selected by the agent.
        """
        obs = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), obs)
        rng, rng_pi = jax.random.split(rng)
        next_actor_rnn_state, pi = self.actor_critic_fn.get_action_distribution(rng_pi, model_state, agent_state, obs)
        rng, rng_sample = jax.random.split(rng)
        action = pi.sample(seed=rng_sample)
        return next_actor_rnn_state, action.squeeze(0), {}

class ActorCriticPPOAgent(RecurrentPPOAgent):
    def __init__(self, actor_critic_fn: BaseActorCriticModel) -> None:
        self.actor_critic_fn = actor_critic_fn
    
    def init_agent_state(self, batch_shape: Sequence[int] = ()) -> PyTreeNode:
        return self.actor_critic_fn.init_rnn_state(batch_shape)
    
    def reset_agent_state(self, agent_state: PyTreeNode, done: jax.Array) -> PyTreeNode:
        default_state = self.init_agent_state(done.shape)
        return jax.tree_util.tree_map(
            lambda x, y: x * (1 - done.reshape(done.shape + (1,) * (x.ndim - done.ndim))) + y, agent_state, default_state
        )
    
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
        obs = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), obs)
        rng, rng_forward = jax.random.split(rng)
        next_agent_state, pi, val = self.actor_critic_fn.forward(rng_forward, model_state, agent_state, obs)
        rng, rng_sample = jax.random.split(rng)
        action = pi.sample(seed=rng_sample)
        log_p = pi.log_prob(action)
        return next_agent_state, action.squeeze(0), {'log_p': log_p.squeeze(0), 'val': val.squeeze(0)}

class ActorBCTrainer(BCTrainer):
    def __init__(
        self,
        actor_critic_fn: BaseActorCriticModel,
        feature_shared: bool = False,
        lr: float = 1e-4,
        grad_clip_norm: float = 10.0,
        num_iterations: int = 1,
        chunk_length: int = 10,
        num_minibatches: int = 1,
        minibatch_num_chunks: int = None
    ) -> None:
        """
        See super().__init__() for more description.

        [Param (additional)]
            lr: Learning rate for the policy (actor) network to fit target data
            grad_clip_norm: Maximum norm for gradient clipping to prevent exploding gradients and improve stability.
        """
        super().__init__(
            num_iterations=num_iterations,
            chunk_length=chunk_length,
            num_minibatches=num_minibatches,
            minibatch_num_chunks=minibatch_num_chunks
        )
        self.actor_critic_fn = actor_critic_fn
        self.feature_shared = feature_shared
        self.bc_config.update({
            'lr': lr,
            'grad_clip_norm': grad_clip_norm
        })
        self.tx = optax.chain(
            optax.clip_by_global_norm(self.bc_config['grad_clip_norm']),
            optax.adam(
                learning_rate=self.bc_config['lr']
            )
        )

    def init_trainer_state(self, rng: jax.Array) -> PyTreeNode:
        """
        Initialize the trainer state.
        """
        if self.feature_shared:
            rng, rng_init_model = jax.random.split(rng)
            model_state = self.actor_critic_fn.init_model_state(rng_init_model)
            actor_critic_train_state = TrainState.create(
                apply_fn=self.actor_critic_fn.actor_fn.apply,
                params=model_state['actor_critic_params'],
                tx=self.tx
            )
            return {'actor_critic_train_state': actor_critic_train_state}
        else:
            rng, rng_init_model = jax.random.split(rng)
            model_state = self.actor_critic_fn.init_model_state(rng_init_model)
            actor_train_state = TrainState.create(
                apply_fn=self.actor_critic_fn.actor_fn.apply,
                params=model_state['actor_params'],
                tx=self.tx
            )
            return {'actor_train_state': actor_train_state}
        
    def model_state_from_trainer_state(self, trainer_state: PyTreeNode) -> PyTreeNode:
        """
        Retreive the model state from the trainer state.
        """
        if self.feature_shared:
            return {
                'actor_critic_params': trainer_state['actor_critic_train_state'].params,
            }
        else:
            return {
                'actor_params': trainer_state['actor_train_state'].params,
            }
    
    def evaluate_target_action(
        self,
        model_state: PyTreeNode,
        obs: Observation,
        target_action: Action,
        aux_data: dict[str, PyTreeNode]
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        [Input]
            - model_state: Model state (e.g., model parameters).
            - obs: Observation with shape [chunk_length, *batch_shape, *observation_shape].
            - target_action: Action to be evaluated with shape [chunk_length, *batch_shape, *action_shape].
            - aux_data: Dictionary containing any additional information needed for computing the log probability and the entropy.
        
        [Output]
            - log_p: Log probability of the taken action with shape [chunk_length, *batch_shape].
            - acc: Accuracy of predicting the target_action with shape [chunk_length, *batch_shape].
        """
        if self.feature_shared:
            rnn_state = {
                'rnn_state': aux_data['agent_state']['rnn_state'][0],
            }
            next_rnn_state, pi = self.actor_critic_fn.get_action_distribution(jax.random.key(0), model_state, rnn_state, obs)
        else:
            rnn_state = {
                'actor_rnn_state': aux_data['agent_state']['rnn_state'][0],
            }
            next_rnn_state, pi = self.actor_critic_fn.get_action_distribution(jax.random.key(0), model_state, rnn_state, obs)
        log_p = pi.log_prob(target_action)
        action_mode = pi.mode()
        eq_dict = {
            k: jnp.isclose(action_mode[k], target_action[k]).reshape(*pi[k].batch_shape, -1).all(-1)
            for k in target_action.keys()
        }
        acc = jnp.stack(list(eq_dict.values()), 0).all(0)
        return log_p, acc
    
    def model_gradient_update(
        self,
        trainer_state: PyTreeNode,
        obs: Observation,
        target_action: Action,
        aux_data: dict[str, PyTreeNode]
    ) -> tuple[PyTreeNode, dict[str, jax.Array]]:
        """
        [Input]
            - model_state: Model state (e.g., model parameters).
            - obs: Observation with shape [chunk_length, *batch_shape, *observation_shape].
            - target_action: Target action  with shape [chunk_length, *batch_shape, *action_shape].
            - aux_data: Dictionary containing any additional information needed for computing the policy loss.
        
        [Output]
            - new_trainer_state: Updated trainer state after one gradient update using the PPO loss.
            - optim_log: Training logs.
        """
        model_state = self.model_state_from_trainer_state(trainer_state)
        (loss, info), grads = jax.value_and_grad(self.nll_loss, has_aux=True)(
            model_state,
            obs=obs,
            target_action=target_action,
            aux_data=aux_data
        )

        if self.feature_shared:
            info['grad_norm'] = global_norm(grads['actor_critic_params'])
            new_actor_critic_train_state = trainer_state['actor_critic_train_state'].apply_gradients(grads=grads['actor_critic_params'])
            return {'actor_critic_train_state': new_actor_critic_train_state}, info
        else:
            info['actor_grad_norm'] = global_norm(grads['actor_params'])
            new_actor_train_state = trainer_state['actor_train_state'].apply_gradients(grads=grads['actor_params'])
            return {'actor_train_state': new_actor_train_state}, info

class ActorCriticPPOTrainer(PPOTrainer):
    def __init__(
        self,
        actor_critic_fn: BaseActorCriticModel,
        feature_shared: bool = False,
        lr: float = None,
        pi_lr: float = None,
        val_lr: float = None,
        lr_schedule_steps: int = None,
        grad_clip_norm: float = 10.0,
        val_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        gamma: float = 0.99,
        gae_lam: float = 0.95,
        ppo_epochs: int = 10,
        ratio_clip: float = 0.2,
        value_clip: float = None,
        use_advantage_normalization: bool = True,
        chunk_length: int = 10,
        num_minibatches: int = 1,
        minibatch_num_chunks: int = None
    ) -> None:
        """
        See super().__init__() for more description.

        [Param (additional)]
            pi_lr: Learning rate for the policy (actor) network.
            val_lr: Learning rate for the value (critic) network.
            grad_clip_norm: Maximum norm for gradient clipping to prevent exploding gradients and improve stability.
            lr_schedule_steps: Number of training steps over which the learning rate is linearly decayed to zero. If None, a constant learning rate will be used.
        """
            
        super().__init__(
            val_loss_coef=val_loss_coef,
            entropy_coef=entropy_coef,
            gamma=gamma,
            gae_lam=gae_lam,
            ppo_epochs=ppo_epochs,
            ratio_clip=ratio_clip,
            value_clip=value_clip,
            use_advantage_normalization=use_advantage_normalization,
            chunk_length=chunk_length,
            num_minibatches=num_minibatches,
            minibatch_num_chunks=minibatch_num_chunks
        )
        self.actor_critic_fn = actor_critic_fn
        self.feature_shared = feature_shared
        self.ppo_config.update({
            'lr': lr,
            'pi_lr': pi_lr if pi_lr is not None else lr,
            'val_lr': val_lr if val_lr is not None else lr,
            'grad_clip_norm': grad_clip_norm,
            'lr_schedule_steps': lr_schedule_steps
        })
        if self.feature_shared:
            self.actor_critic_tx = optax.chain(
                optax.clip_by_global_norm(self.ppo_config['grad_clip_norm']),
                optax.adam(
                    learning_rate=optax.linear_schedule(
                        init_value=self.ppo_config['lr'],
                        end_value=0,
                        transition_steps=self.ppo_config['lr_schedule_steps']
                    )
                ) if self.ppo_config['lr_schedule_steps'] is not None else
                optax.adam(
                    learning_rate=self.ppo_config['lr']
                )
            )
        else:
            self.actor_tx = optax.chain(
                optax.clip_by_global_norm(self.ppo_config['grad_clip_norm']),
                optax.adam(
                    learning_rate=optax.linear_schedule(
                        init_value=self.ppo_config['pi_lr'],
                        end_value=0,
                        transition_steps=self.ppo_config['lr_schedule_steps']
                    )
                ) if self.ppo_config['lr_schedule_steps'] is not None else
                optax.adam(
                    learning_rate=self.ppo_config['pi_lr']
                )
            )
            self.critic_tx = optax.chain(
                optax.clip_by_global_norm(self.ppo_config['grad_clip_norm']),
                optax.adam(
                    learning_rate=optax.linear_schedule(
                        init_value=self.ppo_config['val_lr'],
                        end_value=0,
                        transition_steps=self.ppo_config['lr_schedule_steps']
                    )
                ) if self.ppo_config['lr_schedule_steps'] is not None else
                optax.adam(
                    learning_rate=self.ppo_config['val_lr']
                )
            )
    
    def init_trainer_state(self, rng: jax.Array) -> PyTreeNode:
        """
        Initialize the trainer state.
        """
        if self.feature_shared:
            rng, rng_init_model = jax.random.split(rng)
            model_state = self.actor_critic_fn.init_model_state(rng)
            actor_critic_train_state = TrainState.create(
                apply_fn=self.actor_critic_fn.actor_critic_fn.apply,
                params=model_state['actor_critic_params'],
                tx=self.actor_critic_tx
            )
            return {'actor_critic_train_state': actor_critic_train_state}
        else:
            rng, rng_init_model = jax.random.split(rng)
            model_state = self.actor_critic_fn.init_model_state(rng_init_model)
            actor_train_state = TrainState.create(
                apply_fn=self.actor_critic_fn.actor_fn.apply,
                params=model_state['actor_params'],
                tx=self.actor_tx
            )
            critic_train_state = TrainState.create(
                apply_fn=self.actor_critic_fn.critic_fn.apply,
                params=model_state['critic_params'],
                tx=self.critic_tx
            )
            return {'actor_train_state': actor_train_state, 'critic_train_state': critic_train_state}

    def model_state_from_trainer_state(self, trainer_state: PyTreeNode) -> PyTreeNode:
        """
        Retreive the model state from the trainer state.
        """
        if self.feature_shared:
            return {'actor_critic_params': trainer_state['actor_critic_train_state'].params}
        else:
            return {
                'actor_params': trainer_state['actor_train_state'].params,
                'critic_params': trainer_state['critic_train_state'].params
            }
    
    def evaluate_action(
        self,
        model_state: PyTreeNode,
        obs: Observation,
        action: Action,
        aux_data: dict[str, PyTreeNode]
    ) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, float | jax.Array]]:
        """
        [Input]
            - model_state: Model state (e.g., model parameters).
            - obs: Observation with shape [chunk_length, *batch_shape, *observation_shape].
            - action: Action to be evaluated with shape [chunk_length, *batch_shape, *action_shape].
            - aux_data: Dictionary containing any additional information needed for computing the log probability and the entropy.
        
        [Output]
            - log_p: Log probability of the taken action with shape [chunk_length, *batch_shape].
            - entropy: Entropy of the policy distribution with shape [chunk_length, *batch_shape].
            - value: Estimated value function output with shape [chunk_length, *batch_shape].
            - info: Extra logging/debugging info; values should be floats or scalar jax.Arrays.
        """
        if self.feature_shared:
            rnn_state = {'rnn_state': aux_data['agent_state']['rnn_state'][0]}
            next_rnn_state, pi, val = self.actor_critic_fn.forward(jax.random.key(0), model_state, rnn_state, obs)
        else:
            rnn_state = {
                'actor_rnn_state': aux_data['agent_state']['actor_rnn_state'][0],
                'critic_rnn_state': aux_data['agent_state']['critic_rnn_state'][0]
            }
            next_rnn_state, pi, val = self.actor_critic_fn.forward(jax.random.key(0), model_state, rnn_state, obs)
        log_p = pi.log_prob(action)
        entropy = pi.entropy()
        return log_p, entropy, val, get_distribution_info(pi)

    def model_gradient_update(
        self,
        trainer_state: PyTreeNode,
        buffer: PPOTransition,
        advantage: jax.Array,
        target_val: jax.Array,
        aux_data: dict[str, PyTreeNode]
    ) -> tuple[PyTreeNode, dict[str, jax.Array]]:
        """
        [Input]
            - trainer_state: Trainer state (e.g., model parameters, optimizer states, step, etc.).
            - buffer: PPO transition buffer with attributes having shape [chunk_length, *batch_shape, *attribute_shape].
            - target_val: Target value function with shape [chunk_length, *batch_shape].
            - advantage: Advantage estimates with shape [chunk_length, *batch_shape].
            - aux_data: Dictionary containing any additional information.
        
        [Output]
            - new_trainer_state: Updated trainer state after one gradient update using the PPO loss.
            - optim_log: Training logs.
        """
        model_state = self.model_state_from_trainer_state(trainer_state)
        (loss, info), grads = jax.value_and_grad(self.ppo_loss, has_aux=True)(
            model_state,
            buffer=buffer,
            advantage=advantage,
            target_val=target_val,
            aux_data=aux_data
        )

        if self.feature_shared:
            info['grad_norm'] = global_norm(grads['actor_critic_params'])
            new_actor_critic_train_state = trainer_state['actor_critic_train_state'].apply_gradients(grads=grads['actor_critic_params'])
            return {'actor_critic_train_state': new_actor_critic_train_state}, info
        else:
            info['actor_grad_norm'] = global_norm(grads['actor_params'])
            info['critic_grad_norm'] = global_norm(grads['critic_params'])
            new_actor_train_state = trainer_state['actor_train_state'].apply_gradients(grads=grads['actor_params'])
            new_critic_train_state = trainer_state['critic_train_state'].apply_gradients(grads=grads['critic_params'])
            return {'actor_train_state': new_actor_train_state, 'critic_train_state': new_critic_train_state}, info
