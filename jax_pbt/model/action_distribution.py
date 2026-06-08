from distrax import Distribution
import flax.linen as nn
import jax
import jax.numpy as jnp

from ..env.spaces import Action, ActionSpace, DiscreteSpace, ContinuousSpace
from .nn_blocks.distribution_layers import MultivariateNormalDiag, Categorical, DiagGaussianGlobalVariance, NaiveDisrete


class ActionDistribution(dict[str, Distribution]):
    """
    ActionDistribution represents a dictionary-like structure mapping action attribute names (strings) to corresponding probability distributions (distrax.Distribution instances).

    This class implements some methods from distrax.Distribution, including [sampling, log probability (log_prob), entropy].
    
    Example:
        action_dist = ActionDistribution({
            'continuous': MultivariateNormalDiag(mu, std),
            'discrete': Categorical(logits=logits),
        })
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure all keys are str and all values in the dict are instances of Distribution
        for key, value in self.items():
            if not isinstance(key, str):
                raise ValueError(f"Key '{key}' must be a string, but found type {type(key)}.")
            if not isinstance(value, Distribution):
                raise ValueError(f"Value for key '{key}' must be an instance of Distribution, but found type {type(value)}.")
    
    def keys(self) -> list[str]:
        return sorted(super().keys())

    def values(self) -> list[Distribution]:
        return [self[key] for key in self.keys()]

    def items(self) -> list[tuple[str, Distribution]]:
        return [(key, self[key]) for key in self.keys()]

    def sample(self, seed: jax.Array) -> Action:
        """
        Samples from each distribution in the dictionary using a split RNG.
        """
        seed_split = jax.random.split(seed, len(self.keys()))
        return Action({k: self[k].sample(seed=seed_split[i]) for i, k in enumerate(self.keys())})

    def mode(self) -> Action:
        return Action({k: self[k].mode() for k in self.keys()})
    
    def log_prob(self, action: Action) -> jax.Array:
        return sum([self[k].log_prob(action[k]) for k in self.keys()])

    def entropy(self) -> jax.Array:
        return sum([self[k].entropy() for k in self.keys()])
    
def get_distribution_info(pi: ActionDistribution) -> dict[str, float | jax.Array]:
    info = {}
    for k, d in pi.items():
        if isinstance(d, MultivariateNormalDiag):
            info[f"action[{k}].mu.mean()"] = d._loc.mean()
            info[f"action[{k}].mu.median()"] = jnp.median(d._loc)
            info[f"action[{k}].mu.std()"] = d._loc.std()
            info[f"action[{k}].mu.min()"] = d._loc.min()
            info[f"action[{k}].mu.max()"] = d._loc.max()
            info[f"action[{k}].sigma.mean()"] = d._scale_diag.mean()
            info[f"action[{k}].sigma.median()"] = jnp.median(d._scale_diag)
            info[f"action[{k}].sigma.std()"] = d._scale_diag.std()
            info[f"action[{k}].sigma.min()"] = d._scale_diag.min()
            info[f"action[{k}].sigma.max()"] = d._scale_diag.max()
        elif isinstance(d, Categorical):
            n = d.num_categories
            mode_action = d.mode().flatten()
            counts = jnp.bincount(mode_action, length=n)
            frequency = counts / counts.sum()
            for i in range(min(10, n)):
                info[f"action[{k}].action-{i}-frequency"] = frequency[i]
            top_actions = jnp.argsort(counts)[::-1]
            for i in range(min(3, n)):
                info[f"action[{k}].top-{i + 1}-value"] = top_actions[i]
                info[f"action[{k}].top-{i + 1}-frequency"] = frequency[top_actions[i]]
        else:
            raise TypeError(
                f"Unsupported distribution type for action '{k}': {type(d)}. "
            )
    return info

def build_action_distribution_layer(action_space: ActionSpace) -> dict[str, nn.Module]:
    policy_layers = {}
    for action_name, action_space in action_space.items():
        if isinstance(action_space, ContinuousSpace):
            policy_layers[action_name] = DiagGaussianGlobalVariance(output_shape=action_space.shape)
        elif isinstance(action_space, DiscreteSpace):
            policy_layers[action_name] = NaiveDisrete(output_cardinality=action_space.n)
        else:
            raise TypeError(
                f"Unsupported action space type for '{action_name}': {type(action_space)}. "
            )
    return policy_layers
