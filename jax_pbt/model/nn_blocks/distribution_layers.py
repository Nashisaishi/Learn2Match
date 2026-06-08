from typing import Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class MultivariateNormalDiag(distrax.MultivariateNormalDiag):
    def __repr__(self) -> str:
        loc_shape_repr = ','.join(str(s) for s in self._loc.shape)
        scale_diag_shape_repr = ','.join(str(s) for s in self._scale_diag.shape)
        s = f"[blue]Gaussian[/blue](loc=[dim]{self._loc.dtype}[/dim][{loc_shape_repr}], scale_diag=[dim]{self._scale_diag.dtype}[/dim][{scale_diag_shape_repr}])"
        return s

class Categorical(distrax.Categorical):
    def __repr__(self) -> str:
        logits, probs = self._logits, self._probs
        n = self.num_categories
        shape_repr = ",".join(str(s) for s in (logits.shape if logits is not None else probs.shape))
        dtype = logits.dtype if logits is not None else probs.dtype
        s = f"[blue]Categorical[/blue](n={n}, {'logits' if logits is not None else 'probs'}=[dim]{dtype}[/dim][{shape_repr}])"
        return s

class DiagGaussianGlobalVariance(nn.Module):
    """
    A Gaussian distribution module with a globally shared variance.
    
    Attributes:
        output_shape: The shape of the output distribution mean.
    """
    output_shape: Sequence[int]
    def setup(self):
        self.mu = nn.Dense(np.prod(self.output_shape))
        self.log_sigma = self.param('global_log_sigma', lambda rng, shape: jnp.zeros(shape), self.output_shape)

    def __call__(self, feature: jax.Array) -> MultivariateNormalDiag:
        """
        [Input]
            - feature (jax.Array): Input feature array with shape (..., feature_dim).
        
        [Param]
            - self.mu: A dense layer that produces a flat output of shape (..., np.prod(self.output_shape)).
            - self.log_sigma: A global log standard deviation parameter with shape `self.output_shape`.
        
        [Output]
            - MultivariateNormalDiag: A multivariate normal distribution with mean `mu` 
            reshaped to (..., *self.output_shape) and (diagonal) standard deviation `sigma = exp(self.log_sigma)`.
        """
        mu = self.mu(feature)
        mu = mu.reshape(*mu.shape[:-1], *self.output_shape)
        sigma = jnp.exp(self.log_sigma)
        return MultivariateNormalDiag(mu, sigma)

class NaiveDisrete(nn.Module):
    """
    A simple categorical distribution module.
    
    Attributes:
        output_cardinality (int): The number of discrete categories in the output distribution.
    """
    output_cardinality: int
    def setup(self):
        self.logits = nn.Dense(self.output_cardinality)
    
    def __call__(self, feature: jax.Array) -> Categorical:
        """
        [Input]
            - feature (jax.Array): Input feature array with shape (..., feature_dim).
        
        [Param]
            - self.logits: A dense layer that produces logits of shape (..., self.output_cardinality).
        
        [Output]
            - Categorical: A categorical distribution parameterized by `logits`.
        """
        logits = self.logits(feature)
        return Categorical(logits=logits)



# For flax nn.tabulate
# ---------------------------------------------------------------------
# 1. Teach PyYAML how to serialize distrax distributions
#    (Flax's tabulate uses yaml.safe_dump internally)
# ---------------------------------------------------------------------

from yaml import SafeDumper

def categorical_representer(dumper: SafeDumper, data: Categorical) -> None:
    return dumper.represent_scalar("tag:yaml.org,2002:str", repr(data))

def gaussian_diag_representer(dumper: SafeDumper, data: MultivariateNormalDiag) -> None:
    return dumper.represent_scalar("tag:yaml.org,2002:str", repr(data))

SafeDumper.add_representer(Categorical, categorical_representer)
SafeDumper.add_representer(MultivariateNormalDiag, gaussian_diag_representer)