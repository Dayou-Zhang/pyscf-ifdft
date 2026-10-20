from collections.abc import Iterable

from .backends.libifxc import canonical_feature_set_key
from .ifxc import IFXCEvaluator
from .functionals.ml26 import ML26FeatureEvaluator
from .functionals.mlx26 import MLX26_FEATURE_KEYS
from .mixers.ml25 import ML25Mixer
from .mixers.ml26 import ML26Mixer
from .mixers.mlx26 import MLX26Mixer
from .models import IFModel, IFModelSpec, build_feature_map

_SPECS: dict[str, IFModelSpec] = {}
_MODELS: dict[str, IFModel] = {}
_ALIASES: dict[str, str] = {}


def _normalize_name(name: str) -> str:
    text = str(name).strip().upper()
    if text.startswith("MGGA_XC_"):
        text = text[len("MGGA_XC_") :]
    return text


def _make_backend(feature_set: str):
    if str(feature_set).strip().lower() == "ml26":
        return ML26FeatureEvaluator()
    key = canonical_feature_set_key(feature_set)
    return IFXCEvaluator(key)


def register_functional(spec: IFModelSpec, aliases: Iterable[str] = ()) -> IFModelSpec:
    if not isinstance(spec, IFModelSpec):
        raise TypeError("register_functional expects an IFModelSpec")
    canonical = _normalize_name(spec.name)
    _SPECS[canonical] = spec
    _MODELS.pop(canonical, None)
    _ALIASES[canonical] = canonical
    for alias in aliases:
        _ALIASES[_normalize_name(alias)] = canonical
    return spec


def available_functionals() -> tuple[str, ...]:
    return tuple(sorted(spec.name for spec in _SPECS.values()))


def describe_functional(name: str) -> IFModelSpec:
    key = _ALIASES.get(_normalize_name(name), _normalize_name(name))
    try:
        return _SPECS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown IF functional {name!r}") from exc


def get_functional(name: str) -> IFModel:
    key = _ALIASES.get(_normalize_name(name), _normalize_name(name))
    try:
        model = _MODELS[key]
    except KeyError:
        spec = describe_functional(name)
        backend = _make_backend(spec.feature_set)
        feature_map, inverse_map = build_feature_map(spec, backend)
        model = IFModel(spec=spec, backend=backend, feature_map=feature_map, inverse_map=inverse_map)
        _MODELS[key] = model
    return model


_ml25_mixer = ML25Mixer.from_npz()
register_functional(
    IFModelSpec(
        name="ML25",
        feature_set="ml25",
        mixer=_ml25_mixer,
        hf_coeff=_ml25_mixer.hf_coeff,
        xctype="MGGA",
    ),
    aliases=("MGGA_XC_ML25",),
)

register_functional(
    IFModelSpec(
        name="ML26",
        feature_set="ml26",
        mixer=ML26Mixer.from_npz(),
        hf_coeff=0.0,
        xctype="MGGA",
        supports_scf=False,
        supports_gradients=False,
        supports_response=False,
    ),
    aliases=("MGGA_XC_ML26",),
)

_mlx26_mixer = MLX26Mixer.from_npz()
register_functional(
    IFModelSpec(
        name="MLX26",
        feature_set="ml26",
        mixer=_mlx26_mixer,
        feature_keys=MLX26_FEATURE_KEYS,
        hf_coeff=_mlx26_mixer.hf_coeff,
        xctype="MGGA",
        supports_scf=False,
        supports_gradients=False,
        supports_response=False,
    ),
    aliases=("MGGA_XC_MLX26",),
)
