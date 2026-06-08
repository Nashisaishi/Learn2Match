"""Observations and actions in this training framework are represented as dictionaries that map descriptive strings to JAX arrays. For instance:

import jax.numpy as jnp

observation = Observation({
    'map': jnp.zeros((3, 3), dtype=jnp.float32),
    'symbolic': jnp.zeros((10,), dtype=jnp.float32),
})

action = Action({
    'continuous': jnp.zeros((5,), dtype=jnp.float32),
    'discrete': jnp.zeros((), dtype=jnp.int32),
})


The observation and action spaces for the agents are defined as follows:

observation_space = ObservationSpace({
    'map': (3, 3),
    'symbolic': (10,),
})

action_space = ActionSpace({
    'continuous': (5,),
    'discrete': 10,  # Indicates 10 discrete actions
})

Here, (3, 3), (10,), and (5,) specify the shapes for continuous spaces, while 10 indicates the number of discrete actions. These are equivalent to explicit constructions:

observation_space = ObservationSpace({
    'map': ContinuousSpace((3, 3), low=-1, high=1),
    'symbolic': ContinuousSpace((10,), low=-1, high=1),
})

action_space = ActionSpace({
    'continuous': ContinuousSpace((5,), low=-1, high=1),
    'discrete': DiscreteSpace(10),
})


Environment API

Calling env.get_observation_space() or env.get_action_space() returns (a list of) observation/action space definitions (for each policy/agent) shown above.

Calling env.reset() or env.step() produces (a list of) observations, where each element represents a (batched) observation (for each policy/agent).

See the README for an example of calling the environment API.


Note: the term "agent" or "policy" refers to the entities controlled externally (e.g., by a reinforcement learning algorithm). This differs from "environment agents" (env-agents), which are internal to the environment and used to structure observations/actions. For example:

# Example env-agent observations (usually should not be exposed to external algorithms)
observation = Observation({
    'map': jnp.zeros((5, 3, 3), dtype=jnp.float32),  # 5 env-agents for 'map'
    'symbolic': jnp.zeros((4, 10), dtype=jnp.float32),  # 4 env-agents for 'symbolic'
})

Note that the number of env-agents per attribute (e.g., 5 and 4 above) does not necessarily correspond to the total number of agents in the environment (env.num_agents), as env-agents serve only as internal groupings of attributes.

The internal observation/action space for env-agents can also be queried, providing both the attribute shapes and the number of env-agents per attribute:
env.env_observation_space = (
    ObservationSpace({
        'map': (3, 3),
        'symbolic': (10,),
    }),  # Attribute shapes
    {
        'map': (5,)  # Number (batch shape) of env-agents for 'map'
        'symbolic': (4,),  # Number (batch shape) of env-agents for 'symbolic'
    },
)
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Generic, Mapping, Sequence, TypeVar, Iterator

import jax
import jax.numpy as jnp
import numpy as np


DictedArrayType = TypeVar('DictedArrayType', bound='DictedArray')

class DictedArray(Mapping[str, jax.Array], Generic[DictedArrayType]):
    """
    A dictionary of jax.Array objects with convenient access to common array properties and methods.

    Frequently-used properties are provided, returning dictionaries of these properties for each array in the dictionary.
    
    Properties currently supported are [dtype, ndim, size, shape, at].

    Methods called on the DictedArray are automatically applied to each array in the dictionary if they are callable.
    """
    def __init__(self, array_dict: dict[str, jax.Array]) -> None:
        invalid_keys = [key for key in array_dict.keys() if not isinstance(key, str)]
        if invalid_keys:
            raise TypeError(f"All keys in array_dict must be strings, found {invalid_keys}")
        self._dict = deepcopy(array_dict)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._dict})"

    def __getitem__(self, key: str | int | slice | tuple[int | slice]) -> jax.Array | DictedArrayType:
        def valid_index(x):
            return isinstance(x, (int, slice, type(Ellipsis))) or isinstance(x, (jax.Array, np.ndarray)) and (jnp.issubdtype(x.dtype, jnp.integer) or jnp.issubdtype(x.dtype, jnp.bool))
        if isinstance(key, str) and key in self._dict:
            return self._dict[key]
        elif valid_index(key) or isinstance(key, tuple) and all(valid_index(x) for x in key):
            return self.__class__({k: v[key] for k, v in self.items()})
        else:
            raise TypeError(
                f"Index of {key} with type {type(key)} is not supported for {type(self)}."
            )

    def __setitem__(self, key: str, value: jax.Array) -> None:
        self._dict[key] = value

    def __delitem__(self, key: str) -> None:
        del self._dict[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def keys(self) -> list[str]:
        return sorted(self._dict.keys())

    def values(self) -> list[jax.Array]:
        return [self._dict[key] for key in self.keys()]

    def items(self) -> list[tuple[str, jax.Array]]:
        return [(key, self._dict[key]) for key in self.keys()]
    
    @property
    def dtype(self) -> dict[str, jnp.dtype]:
        return {k: v.dtype for k, v in self.items()} 

    @property
    def ndim(self) -> dict[str, int]:
        return {k: v.ndim for k, v in self.items()} 

    @property
    def size(self) -> dict[str, int]:
        return {k: v.size for k, v in self.items()} 

    @property
    def shape(self) -> dict[str, tuple[int, ...]]:
        return {k: v.shape for k, v in self.items()} 

    @property
    def at(self) -> 'DictedArrayType':
        return self.__class__({k: v.at for k, v in self.items()})

    def set(self, values: DictedArrayType) -> DictedArrayType:
        return self.__class__({k: v.set(values[k]) for k, v in self.items()})
    
    @classmethod
    def concatenate(cls, arrays: Sequence[DictedArrayType], axis: int = 0) -> DictedArrayType:
        return cls({
            k: jnp.concatenate([v[k] for v in arrays], axis=axis)
            for k in arrays[0].keys()
        })

    @classmethod
    def stack(cls, arrays: Sequence[DictedArrayType], axis: int = 0) -> DictedArrayType:
        return cls({
            k: jnp.stack([v[k] for v in arrays], axis=axis)
            for k in arrays[0].keys()
        })
    
    @classmethod
    def split(cls, ary: DictedArrayType, indices_or_sections: int | Sequence[int], axis=0) -> list[DictedArrayType]:
        split_dict_of_lists = {k: jnp.split(v, indices_or_sections, axis=axis) for k, v in ary.items()}
        n = indices_or_sections if isinstance(indices_or_sections, int) else len(indices_or_sections) + 1
        return [
            cls({k: split_dict_of_lists[k][i] for k in ary.keys()})
            for i in range(n)
        ]
    
    @classmethod
    def batch_flatten(cls, ary: DictedArrayType, batch_shape: Sequence[int]) -> DictedArrayType:
        # flatten the multi-dimensional shape into one dimension
        batch_size = int(np.prod(batch_shape))
        return cls({k: v.reshape(batch_size, *v.shape[len(batch_shape):]) for k, v in ary.items()})
    
    @classmethod
    def batch_unflatten(cls, ary: DictedArrayType, batch_shape: Sequence[int]) -> DictedArrayType:
        # batch_shape is the multi-dimensional shape before flatten()
        return cls({
            k: v.reshape(*batch_shape, *v.shape[1:])
            for k, v in ary.items()
        })

    def __getattr__(self, name: str):
        """
        Dynamically handle frequently-used method calls for jax.Array objects in the dictionary. For supprted methods, all attributes should share the same input parameters.

        If the attribute with the given name is a callable method on the jax.Array objects,
        returns a function that applies this method to each array in the dictionary and returns a new DictedArray with the results.

        Only callable methods are supported automatically. The methods usually cannot take jax.Array objects as input parameters.
        """
        # Collect the specified attribute from each jax.Array in the dictionary
        if name not in ['all', 'any', 'argmax', 'argmin', 'argpartition', 'argsort', 'astype', 'conj', 'conjugate', 'copy', 'copy_to_host_async', 'cumprod', 'cumsum', 'diagonal', 'flat', 'flatten', 'max', 'mean', 'min', 'ravel', 'repeat', 'reshape', 'round', 'shape', 'sort', 'squeeze', 'std', 'sum', 'swapaxes', 'trace', 'transpose', 'var', 'view', 'T', 'mT']:
            return super().__getattr__(name)
        
        # Return a method that applies the callable to each jax.Array
        def method(*args, **kwargs) -> 'DictedArrayType':
            return self.__class__({k: getattr(v, name)(*args, **kwargs) for k, v in self.items()})
        return method
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._dict})"
    
    def tree_flatten(self):
        aux_data = tuple(sorted(self._dict.keys()))
        children = tuple(self._dict[k] for k in aux_data)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(dict(zip(aux_data, children)))
    
    def raw(self, in_place=False) -> dict[str, jax.Array]:
        if in_place:
            return self._dict
        else:
            return jax.tree_util.tree_map(jnp.copy, self._dict)


# Define types for the observation and action structures

@jax.tree_util.register_pytree_node_class
class Observation(DictedArray['Observation']):
    pass

@jax.tree_util.register_pytree_node_class
class Action(DictedArray['Action']):
    pass

# BaseSpace instances, used to define different single-modal observation/action space
class BaseSpace(ABC):
    @abstractmethod
    def example(self, batch_shape: Sequence[int] = ()) -> jax.Array:
        ...

    @abstractmethod
    def sample(self, rng: jax.Array) -> jax.Array:
        ...
    
    @abstractmethod
    def __eq__(self, other) -> bool:
        ...

    @abstractmethod
    def __repr__(self) -> str:
        ...

SpaceType = TypeVar('SpaceType', bound='BaseSpace')

class DiscreteSpace(BaseSpace):
    def __init__(self, n: int) -> None:
        if not isinstance(n, int):
            raise TypeError(f"'n' must be an integer, got {type(n).__name__}")
        if n < 0:
            raise ValueError(f"'n' must be non-negative, got {n}")
        self.n = n

    def example(self, batch_shape: Sequence[int] = ()) -> jax.Array:
        return jnp.zeros(batch_shape, dtype=int)

    def sample(self, key: jax.Array, batch_shape: Sequence[int] = ()) -> jax.Array:
        return jax.random.randint(key, shape=batch_shape, minval=0, maxval=self.n)
    
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, DiscreteSpace) and self.n == other.n
    
    def __repr__(self) -> str:
        return f"DiscreteSpace(n={self.n})"

class ContinuousSpace(BaseSpace):
    def __init__(
        self, shape: Sequence[int], low: float | jax.Array, high: float | jax.Array
    ) -> None:
        if jnp.isscalar(low):
            low = low * jnp.ones(shape)
        if jnp.isscalar(high):
            high = high * jnp.ones(shape)

        if not jnp.array_equal(jnp.all(high >= low), jnp.array(True)):
            raise ValueError(
                f"All elements of 'high': {high} must be greater than or equal to 'low': {low} in ContinuousSpace"
            )

        if low.shape != tuple(shape):
            raise ValueError(
                f"The shape of 'low': {low.shape} must match 'shape:' {shape} in ContinuousSpace"
            )

        if high.shape != tuple(shape):
            raise ValueError(
                f"The shape of 'high': {high.shape} must match 'shape: {shape}' in ContinuousSpace"
            )

        self.shape = tuple(shape)
        self.low = low
        self.high = high

    def example(self, batch_shape: Sequence[int] = ()) -> jax.Array:
        return jnp.copy(jnp.broadcast_to(self.low, (*batch_shape, *self.shape))).astype(jnp.float32)

    def sample(self, key: jax.Array, batch_shape: Sequence[int] = ()) -> jax.Array:
        return jax.random.uniform(key, (*batch_shape, *self.shape), minval=self.low, maxval=self.high).astype(jnp.float32)
    
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, ContinuousSpace) and self.shape == other.shape

    def __repr__(self) -> str:
        return f"ContinuousSpace(shape={self.shape}, " + \
            f"   low={self.low.mean() if jnp.isclose(self.low.min(), self.low.max()) else f'{self.low.mean()} (mean)'}, " + \
            f"   high={self.high.mean() if jnp.isclose(self.high.min(), self.high.max()) else f'{self.high.mean()} (mean)'})"


"""A Space represents a collection of BaseSpace instances for different components of an observation or action.

It acts like a dictionary (or mapping from string to BaseSpace) where each key corresponds to a component name, and each value is a BaseSpace object defining the space of that component.
"""

class Space(Mapping[str, SpaceType]):
    def __init__(self, space_dict: dict[str, int | Sequence[int] | BaseSpace]) -> None:
        invalid_keys = [key for key in space_dict.keys() if not isinstance(key, str)]
        if invalid_keys:
            raise TypeError(f"All keys in space_dict must be strings, found {invalid_keys}")
        self._dict = deepcopy(space_dict)
        for k, v in space_dict.items():
            if isinstance(v, int):
                self._dict[k] = DiscreteSpace(v)
            elif isinstance(v, Sequence):
                # Check that all elements in the sequence are integers
                if all(isinstance(elem, int) for elem in v):
                    # Convert sequences of integers to ContinuousSpace instances
                    self._dict[k] = ContinuousSpace(v, low=-1, high=1)
                else:
                    raise ValueError(
                        f"All elements in the sequence for key '{k}' must be integers; found {v}."
                    )
            else:
                # Ensure that the value is an instance of BaseSpace
                assert isinstance(v, BaseSpace), (
                    f"Each space should be a BaseSpace instance; found {v} of type {type(v)} instead."
                )

    def __getitem__(self, key: str) -> BaseSpace:
        return self._dict[key]

    def __setitem__(self, key: str, value: BaseSpace) -> None:
        self._dict[key] = value

    def __delitem__(self, key: str) -> None:
        del self._dict[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def keys(self) -> list[str]:
        return sorted(self._dict.keys())

    def values(self) -> list[BaseSpace]:
        return [self._dict[key] for key in self.keys()]

    def items(self) -> list[tuple[str, BaseSpace]]:
        return [(key, self._dict[key]) for key in self.keys()]

    def example(self, batch_shape: Sequence[int] | dict[str, Sequence[int]]= ()) -> dict[str, jax.Array]:
        if isinstance(batch_shape, dict):
            return {
                k: v.example(batch_shape[k]) for k, v in self.items()
            }
        else:
            return {
                k: v.example(batch_shape) for k, v in self.items()
            }

    def sample(self, key: jax.Array, batch_shape: Sequence[int] | dict[str, Sequence[int]]= ()) -> dict[str, jax.Array]:
        key_split = jax.random.split(key, len(self.keys()))
        if isinstance(batch_shape, dict):
            return {
                k: self[k].sample(key_split[i], batch_shape[k])
                for i, k in enumerate(self.keys())
            }
        else:
            return {
                k: self[k].sample(key_split[i], batch_shape)
                for i, k in enumerate(self.keys())
            }
    
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Space) and self.items() == other.items()
    
    def __repr__(self) -> str:
        items_str = ',\n  '.join(f"{repr(k)}: {repr(v)}" for k, v in self.items())
        return self.__class__.__name__ + '({\n  ' +  items_str + '\n})'

class ObservationSpace(Space):
    def example(self, batch_shape: Sequence[int] | dict[str, Sequence[int]]= ()) -> Observation:
        return Observation(super().example(batch_shape))

class ActionSpace(Space):
    def example(self, batch_shape: Sequence[int] | dict[str, Sequence[int]]= ()) -> Observation:
        return Action(super().example(batch_shape))
