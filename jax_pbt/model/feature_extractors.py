from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from ..env.spaces import Observation, ObservationSpace, ContinuousSpace
from .nn_blocks import Activation, MLP, RNN
from .nn_configs import NNConfig, CNNConfig, MLPConfig


class EmbeddingExtractor(nn.Module):
    nn_configs: dict[str, NNConfig]
    use_embedding_layer_norm: bool = True
    # Keys whose observation carries a per-element "set" axis (e.g. hat_y is
    # (..., Nf, d)). Determined from the obs-space shape at build time so the
    # split is independent of how many leading batch axes are present at call
    # time (rollout adds a time axis on top of the agent-batch axis).
    set_keys: tuple[str, ...] = ()

    # TODO: brief comments
    def setup(self) -> None:
        self.feature_extractors = {
            k: self.nn_configs[k].create()
            for k in sorted(self.nn_configs.keys())
        }
        if self.use_embedding_layer_norm:
            self.layer_norms = {
                k: nn.LayerNorm() for k in sorted(self.nn_configs.keys())
            }
    
    def __call__(
        self, obs: Observation
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        embedding_dict = {
            k: self.feature_extractors[k](obs[k])
            for k in sorted(set(obs.keys()) & set(self.feature_extractors.keys()))
        }
        if self.use_embedding_layer_norm:
            embedding_dict = {k: self.layer_norms[k](v) for k, v in embedding_dict.items()}
        # "Set" observations carry a per-element axis (e.g. hat_y is (..., Nf, d));
        # keep their embeddings separate from the flat global embeddings so the
        # downstream policy can score each element rather than a flattened blob.
        set_keys = set(self.set_keys)
        flat_embeddings = {k: v for k, v in embedding_dict.items() if k not in set_keys}
        set_embeddings = {k: v for k, v in embedding_dict.items() if k in set_keys}
        return flat_embeddings, set_embeddings

class RecurrentFeatureExtractor(nn.Module):
    embedding_extractor_config: dict[str, NNConfig]
    set_keys: tuple[str, ...] = ()
    use_embedding_layer_norm: bool = True
    aggr_mlp_layers: int = 0
    aggr_mlp_hidden_size: int = 64
    aggr_mlp_activation_type: Activation.Type = 'relu'
    use_rnn: bool = False
    rnn_hidden_layers: int = 1
    rnn_hidden_size: int = 64
    use_rnn_layer_norm: bool = True
    use_final_layer_norm: bool = False
        # TODO: a brief and intuitive nn architecuture description
    
    def setup(self) -> None:
        self.embedding_extractor = EmbeddingExtractor(
            nn_configs=self.embedding_extractor_config,
            use_embedding_layer_norm=self.use_embedding_layer_norm,
            set_keys=self.set_keys,
        )
        if self.aggr_mlp_layers > 0:
            self.aggr_mlp = MLP(
                hidden_size_lst=[self.aggr_mlp_hidden_size for i in range(self.aggr_mlp_layers)],
                activation_type=self.aggr_mlp_activation_type
            )
        if self.use_rnn:
            self.rnn_model = RNN(
                rnn_layers=self.rnn_hidden_layers,
                hidden_size=self.rnn_hidden_size,
                use_layer_norm=self.use_rnn_layer_norm
            )
        if self.use_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm()

    def __call__(
        self, rnn_state: jax.Array, obs: Observation
    ) -> tuple[jax.Array, jax.Array, jax.Array | None]:
        flat_dict, set_dict = self.embedding_extractor(obs)
        # Only the flat global embeddings are concatenated into the recurrent
        # feature; the per-element "set" embeddings are returned untouched so the
        # policy can score each element against the global feature.
        feature = jnp.concatenate([flat_dict[k] for k in sorted(flat_dict)], axis=-1)
        firm_emb = set_dict[sorted(set_dict)[0]] if set_dict else None
        if self.aggr_mlp_layers > 0:
            feature = self.aggr_mlp(feature)
        if self.use_rnn:
            rnn_state, feature = self.rnn_model(rnn_state, feature)
        if self.use_final_layer_norm:
            feature = self.final_layer_norm(feature)
        return rnn_state, feature, firm_emb

def build_feature_extractor_with_consistent_layers(
    obs_space: ObservationSpace, 
    hidden_size: int = 64,
    mlp_hidden_layer: int = 2,
    cnn_feature_lst: Sequence[int] = [64, 64, 64], 
    cnn_kernel_lst: Sequence[Sequence[int]] = [(3, 3), (3, 3), (3, 3)], 
    aggr_mlp_layers: int = 0,
    rnn_hidden_layers: int = 1,
    rnn_hidden_size: int = 64,
    use_rnn: bool = False,
    verbose_display: bool = False 
) -> RecurrentFeatureExtractor:
    """
    Loads a feature extractor while ensuring consistent hidden layers 
    across embbedings/feature extraction/aggregation. Uses default hyperparameters where applicable.

    [Input]
        - obs_space: A dictionary mapping observation keys to their respective spaces.
        - hidden_size: Hidden size used for MLP layers, RNN hidden states, and aggregation.
        - mlp_hidden_layer: Number of hidden layers in the MLP configuration.
        - cnn_feature_lst: List of feature sizes for CNN layers.
        - cnn_kernel_lst: List of kernel sizes for CNN layers.
        - aggr_mlp_layers: Number of hidden layers in the aggregation MLP.
        - rnn_hidden_layers: Number of layers in the RNN model.
        - rnn_hidden_size: Hidden size for RNN states.
        - use_rnn: Whether to use an RNN-based feature extractor.
        - verbose_display: If True, prints readable messages for users.

    [Output]
        - RecurrentFeatureExtractor: A feature extractor with consistent hidden layers.
    """
    model_config: dict[str, NNConfig] = {}
    # Keys whose space keeps a per-element axis (2-D symbolic space, e.g. an
    # (Nf, d) set of firm embeddings) are "set" inputs; the rest are flat.
    set_keys = tuple(
        k for k, space in obs_space.items()
        if isinstance(space, ContinuousSpace) and len(space.shape) == 2
    )

    for k, space in obs_space.items():
        if not isinstance(space, ContinuousSpace):
            raise NotImplementedError(
                f"Expected symbolic or pixel observations, but got obs[{k}] = {space}"
            )

        if len(space.shape) == 1:
            if verbose_display:
                print(f"Creating MLPConfig for symbolic input: {k}")
            model_config[k] = MLPConfig(hidden_layers=mlp_hidden_layer, hidden_size=hidden_size)
        elif len(space.shape) == 2:
            if verbose_display:
                print(f"Creating per-element MLPConfig for set input: {k}")
            model_config[k] = MLPConfig(hidden_layers=mlp_hidden_layer, hidden_size=hidden_size)
        elif len(space.shape) == 3:
            if verbose_display:
                print(f"Creating CNNConfig for pixel input: {k}")
            model_config[k] = CNNConfig(
                feature_lst=cnn_feature_lst, 
                kernel_lst=cnn_kernel_lst, 
                post_mlp_hidden_size=hidden_size
            )
        else:
            raise NotImplementedError(f"Unsupported observation shape {space.shape} for obs[{k}]")

    if verbose_display:
        print("Using RecurrentFeatureExtractor with consistent hidden sizes.")

    return RecurrentFeatureExtractor(
        embedding_extractor_config=model_config,
        set_keys=set_keys,
        use_rnn=use_rnn,
        rnn_hidden_layers=rnn_hidden_layers, 
        rnn_hidden_size=rnn_hidden_size,
        aggr_mlp_layers=aggr_mlp_layers,
        aggr_mlp_hidden_size=hidden_size
    )
