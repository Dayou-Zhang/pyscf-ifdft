from __future__ import annotations

from dataclasses import dataclass
import ctypes

import numpy as np

from pyscf import lib

from .exceptions import IFDFTError


class IFDFTBackendError(IFDFTError):
    pass


_WANT_INTEGRAL_FEATURES = 0x01
_WANT_LOCAL_VALUES = 0x02
_WANT_FIRST_DERIVATIVES = 0x04
_WANT_SECOND_DERIVATIVES = 0x08


@dataclass(frozen=True)
class IFXCBlock:
    features: np.ndarray | None
    local_values: np.ndarray | None
    d_rho: np.ndarray | None
    d_sigma: np.ndarray | None
    d_tau: np.ndarray | None
    dd_rho_rho: np.ndarray | None
    dd_rho_sigma: np.ndarray | None
    dd_rho_tau: np.ndarray | None
    dd_sigma_sigma: np.ndarray | None
    dd_sigma_tau: np.ndarray | None
    dd_tau_tau: np.ndarray | None


_lib = lib.load_library("libifdft_ifxc")


def _configure_signatures() -> None:
    _lib.IFDFT_ifxc_version.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    _lib.IFDFT_ifxc_version.restype = ctypes.c_int

    _lib.IFDFT_ifxc_feature_set_count.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
    _lib.IFDFT_ifxc_feature_set_count.restype = ctypes.c_int

    _lib.IFDFT_ifxc_feature_set_key.argtypes = [
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    _lib.IFDFT_ifxc_feature_set_key.restype = ctypes.c_int

    _lib.IFDFT_ifxc_feature_count.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    _lib.IFDFT_ifxc_feature_count.restype = ctypes.c_int

    _lib.IFDFT_ifxc_feature_key.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    _lib.IFDFT_ifxc_feature_key.restype = ctypes.c_int

    _lib.IFDFT_ifxc_feature_set_max_deriv_order.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    _lib.IFDFT_ifxc_feature_set_max_deriv_order.restype = ctypes.c_int

    _lib.IFDFT_ifxc_feature_set_xctype.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    _lib.IFDFT_ifxc_feature_set_xctype.restype = ctypes.c_int

    _lib.IFDFT_ifxc_evaluator_create.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    _lib.IFDFT_ifxc_evaluator_create.restype = ctypes.c_int

    _lib.IFDFT_ifxc_evaluator_destroy.argtypes = [ctypes.c_void_p]
    _lib.IFDFT_ifxc_evaluator_destroy.restype = ctypes.c_int

    _lib.IFDFT_ifxc_evaluator_set_num_threads.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    _lib.IFDFT_ifxc_evaluator_set_num_threads.restype = ctypes.c_int

    double_p = ctypes.POINTER(ctypes.c_double)
    _lib.IFDFT_ifxc_evaluator_eval_block.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        double_p,
        double_p,
        double_p,
        double_p,
        ctypes.c_uint,
        double_p,
        double_p,
        double_p,
        double_p,
        double_p,
        double_p,
        double_p,
        double_p,
        double_p,
        double_p,
        double_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _lib.IFDFT_ifxc_evaluator_eval_block.restype = ctypes.c_int

    _lib.IFDFT_ifxc_evaluator_eval_weighted_vvars.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        double_p,
        double_p,
        double_p,
        double_p,
        double_p,
        double_p,
        double_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _lib.IFDFT_ifxc_evaluator_eval_weighted_vvars.restype = ctypes.c_int

    _lib.IFDFT_ifxc_evaluator_eval_weighted_vvars_from_pyscf_mgga.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        double_p,
        double_p,
        ctypes.c_int,
        double_p,
        double_p,
        double_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _lib.IFDFT_ifxc_evaluator_eval_weighted_vvars_from_pyscf_mgga.restype = ctypes.c_int

    _lib.IFDFT_ifxc_evaluator_eval_weighted_vvars_batch_from_pyscf_mgga.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_size_t,
        double_p,
        double_p,
        ctypes.c_int,
        double_p,
        double_p,
        double_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _lib.IFDFT_ifxc_evaluator_eval_weighted_vvars_batch_from_pyscf_mgga.restype = ctypes.c_int

    _lib.IFDFT_ifxc_evaluator_eval_features_from_pyscf_mgga.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        double_p,
        double_p,
        ctypes.c_int,
        double_p,
        double_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _lib.IFDFT_ifxc_evaluator_eval_features_from_pyscf_mgga.restype = ctypes.c_int

    _lib.IFDFT_ifxc_evaluator_eval_feature_directional_from_pyscf_mgga.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        double_p,
        double_p,
        double_p,
        double_p,
        ctypes.c_int,
        double_p,
        double_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _lib.IFDFT_ifxc_evaluator_eval_feature_directional_from_pyscf_mgga.restype = ctypes.c_int

    _lib.IFDFT_ifxc_evaluator_eval_weighted_response_from_pyscf_mgga.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        double_p,
        double_p,
        double_p,
        double_p,
        ctypes.c_int,
        double_p,
        double_p,
        double_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _lib.IFDFT_ifxc_evaluator_eval_weighted_response_from_pyscf_mgga.restype = ctypes.c_int

    _lib.IFDFT_ifxc_evaluator_eval_response_batch_from_pyscf_mgga.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_size_t,
        double_p,
        double_p,
        double_p,
        double_p,
        ctypes.c_int,
        double_p,
        double_p,
        double_p,
        double_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _lib.IFDFT_ifxc_evaluator_eval_response_batch_from_pyscf_mgga.restype = ctypes.c_int

    _lib.IFDFT_ifxc_evaluator_eval_feature_directional_derivative.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        double_p,
        double_p,
        double_p,
        double_p,
        double_p,
        double_p,
        double_p,
        double_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _lib.IFDFT_ifxc_evaluator_eval_feature_directional_derivative.restype = ctypes.c_int

    _lib.IFDFT_ifxc_evaluator_eval_weighted_response_vvars.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        double_p,
        double_p,
        double_p,
        double_p,
        double_p,
        double_p,
        double_p,
        double_p,
        double_p,
        double_p,
        double_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _lib.IFDFT_ifxc_evaluator_eval_weighted_response_vvars.restype = ctypes.c_int


_configure_signatures()


def _raise_on_status(status: int, errbuf: ctypes.Array[ctypes.c_char] | None = None) -> None:
    if status == 0:
        return
    message = ""
    if errbuf is not None:
        message = bytes(errbuf.value).decode("utf-8", errors="replace")
    if not message:
        message = f"libifxc bridge call failed with status {status}"
    raise IFDFTBackendError(message)


def _version() -> tuple[int, int, int]:
    major = ctypes.c_int()
    minor = ctypes.c_int()
    patch = ctypes.c_int()
    _raise_on_status(_lib.IFDFT_ifxc_version(
        ctypes.byref(major),
        ctypes.byref(minor),
        ctypes.byref(patch),
    ))
    return int(major.value), int(minor.value), int(patch.value)


def _encode_key(feature_set_key: str) -> bytes:
    return str(feature_set_key).encode("utf-8")


def _feature_set_keys() -> tuple[str, ...]:
    count = ctypes.c_size_t()
    _raise_on_status(_lib.IFDFT_ifxc_feature_set_count(ctypes.byref(count)))
    keys = []
    for idx in range(int(count.value)):
        key = ctypes.c_char_p()
        _raise_on_status(_lib.IFDFT_ifxc_feature_set_key(ctypes.c_size_t(idx), ctypes.byref(key)))
        if key.value is None:
            raise IFDFTBackendError(f"libifxc returned a null feature-set key at index {idx}")
        keys.append(key.value.decode("utf-8"))
    return tuple(keys)


def _feature_count(feature_set_key: str) -> int:
    count = ctypes.c_size_t()
    _raise_on_status(_lib.IFDFT_ifxc_feature_count(_encode_key(feature_set_key), ctypes.byref(count)))
    return int(count.value)


def _feature_keys(feature_set_key: str) -> tuple[str, ...]:
    keys = []
    for idx in range(_feature_count(feature_set_key)):
        key = ctypes.c_char_p()
        _raise_on_status(
            _lib.IFDFT_ifxc_feature_key(_encode_key(feature_set_key), ctypes.c_size_t(idx), ctypes.byref(key))
        )
        if key.value is None:
            raise IFDFTBackendError(f"libifxc returned a null feature key at index {idx}")
        keys.append(key.value.decode("utf-8"))
    return tuple(keys)


def _feature_set_max_deriv_order(feature_set_key: str) -> int:
    max_order = ctypes.c_int()
    _raise_on_status(
        _lib.IFDFT_ifxc_feature_set_max_deriv_order(_encode_key(feature_set_key), ctypes.byref(max_order))
    )
    return int(max_order.value)


def _feature_set_xctype(feature_set_key: str) -> str:
    xctype = ctypes.c_char_p()
    _raise_on_status(_lib.IFDFT_ifxc_feature_set_xctype(_encode_key(feature_set_key), ctypes.byref(xctype)))
    if xctype.value is None:
        raise IFDFTBackendError(f"libifxc returned a null xctype for feature set {feature_set_key!r}")
    return xctype.value.decode("utf-8")


def _create_evaluator(feature_set_key: str) -> ctypes.c_void_p:
    handle = ctypes.c_void_p()
    _raise_on_status(_lib.IFDFT_ifxc_evaluator_create(_encode_key(feature_set_key), ctypes.byref(handle)))
    if not handle.value:
        raise IFDFTBackendError(f"libifxc returned a null evaluator for feature set {feature_set_key!r}")
    return handle


def _handle_value(handle: ctypes.c_void_p | int) -> int:
    if isinstance(handle, ctypes.c_void_p):
        if handle.value is None:
            raise IFDFTBackendError("null libifxc evaluator handle")
        return int(handle.value)
    return int(handle)


def _destroy_evaluator(handle: ctypes.c_void_p | int | None) -> None:
    if handle is None:
        return
    _raise_on_status(_lib.IFDFT_ifxc_evaluator_destroy(ctypes.c_void_p(_handle_value(handle))))


def _sync_num_threads(handle: ctypes.c_void_p | int) -> None:
    _raise_on_status(
        _lib.IFDFT_ifxc_evaluator_set_num_threads(
            ctypes.c_void_p(_handle_value(handle)),
            ctypes.c_int(int(lib.num_threads())),
        )
    )


def _spin_dims(spin: int) -> tuple[int, int, int]:
    if spin == 0:
        return 1, 1, 1
    if spin == 1:
        return 2, 3, 2
    raise ValueError(f"Unsupported spin {spin}")


def _sym_count(ncomp: int) -> int:
    return ncomp * (ncomp + 1) // 2


def _as_input(values, ncomp: int, npoints: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64, order="C")
    if arr.shape != (ncomp, npoints):
        raise ValueError(f"{name} must have shape {(ncomp, npoints)}, got {arr.shape}")
    return arr


def _as_weights(weights, npoints: int) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float64, order="C")
    if arr.shape != (npoints,):
        raise ValueError(f"weights must have shape {(npoints,)}, got {arr.shape}")
    return arr


def _ptr(arr: np.ndarray | None):
    if arr is None:
        return None
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _reshape(arr: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray | None:
    if arr is None:
        return None
    return arr.reshape(shape)


def _as_output(out, shape: tuple[int, ...], name: str) -> np.ndarray:
    if out is None:
        return np.empty(shape, dtype=np.float64)
    arr = np.asarray(out)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if arr.dtype != np.float64:
        raise ValueError(f"{name} must have dtype float64, got {arr.dtype}")
    if not arr.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return arr


def _eval_evaluator_block(
    handle: ctypes.c_void_p | int,
    nfeatures: int,
    rho,
    sigma,
    tau,
    weights=None,
    *,
    spin: int,
    want_integral_features: bool = False,
    want_local_values: bool = False,
    want_first_derivatives: bool = False,
    want_second_derivatives: bool = False,
) -> IFXCBlock:
    rho_dim, sigma_dim, tau_dim = _spin_dims(spin)
    rho_arr = np.asarray(rho, dtype=np.float64, order="C")
    if rho_arr.ndim != 2 or rho_arr.shape[0] != rho_dim:
        raise ValueError(f"rho must have shape ({rho_dim}, npoints), got {rho_arr.shape}")
    npoints = int(rho_arr.shape[1])
    sigma_arr = _as_input(sigma, sigma_dim, npoints, "sigma")
    tau_arr = _as_input(tau, tau_dim, npoints, "tau")

    flags = 0
    if want_integral_features:
        flags |= _WANT_INTEGRAL_FEATURES
    if want_local_values:
        flags |= _WANT_LOCAL_VALUES
    if want_first_derivatives:
        flags |= _WANT_FIRST_DERIVATIVES
    if want_second_derivatives:
        flags |= _WANT_SECOND_DERIVATIVES

    weighted = want_integral_features or want_first_derivatives or want_second_derivatives
    weights_arr = _as_weights(weights, npoints) if weighted else None
    nfeatures = int(nfeatures)

    features = np.empty(nfeatures, dtype=np.float64) if want_integral_features else None
    local_values = np.empty((npoints, nfeatures), dtype=np.float64) if want_local_values else None

    d_rho = np.empty((npoints, nfeatures, rho_dim), dtype=np.float64) if want_first_derivatives else None
    d_sigma = np.empty((npoints, nfeatures, sigma_dim), dtype=np.float64) if want_first_derivatives else None
    d_tau = np.empty((npoints, nfeatures, tau_dim), dtype=np.float64) if want_first_derivatives else None

    dd_rho_rho = (
        np.empty((npoints, nfeatures, _sym_count(rho_dim)), dtype=np.float64)
        if want_second_derivatives else None
    )
    dd_rho_sigma = (
        np.empty((npoints, nfeatures, rho_dim * sigma_dim), dtype=np.float64)
        if want_second_derivatives else None
    )
    dd_rho_tau = (
        np.empty((npoints, nfeatures, rho_dim * tau_dim), dtype=np.float64)
        if want_second_derivatives else None
    )
    dd_sigma_sigma = (
        np.empty((npoints, nfeatures, _sym_count(sigma_dim)), dtype=np.float64)
        if want_second_derivatives else None
    )
    dd_sigma_tau = (
        np.empty((npoints, nfeatures, sigma_dim * tau_dim), dtype=np.float64)
        if want_second_derivatives else None
    )
    dd_tau_tau = (
        np.empty((npoints, nfeatures, _sym_count(tau_dim)), dtype=np.float64)
        if want_second_derivatives else None
    )

    _sync_num_threads(handle)
    errbuf = ctypes.create_string_buffer(512)
    status = _lib.IFDFT_ifxc_evaluator_eval_block(
        ctypes.c_void_p(_handle_value(handle)),
        int(spin),
        ctypes.c_size_t(npoints),
        _ptr(rho_arr),
        _ptr(sigma_arr),
        _ptr(tau_arr),
        _ptr(weights_arr),
        ctypes.c_uint(flags),
        _ptr(features),
        _ptr(local_values),
        _ptr(d_rho),
        _ptr(d_sigma),
        _ptr(d_tau),
        _ptr(dd_rho_rho),
        _ptr(dd_rho_sigma),
        _ptr(dd_rho_tau),
        _ptr(dd_sigma_sigma),
        _ptr(dd_sigma_tau),
        _ptr(dd_tau_tau),
        errbuf,
        ctypes.c_size_t(len(errbuf)),
    )
    _raise_on_status(status, errbuf)

    return IFXCBlock(
        features=features,
        local_values=_reshape(local_values, (npoints, nfeatures)),
        d_rho=_reshape(d_rho, (npoints, nfeatures, rho_dim)),
        d_sigma=_reshape(d_sigma, (npoints, nfeatures, sigma_dim)),
        d_tau=_reshape(d_tau, (npoints, nfeatures, tau_dim)),
        dd_rho_rho=_reshape(dd_rho_rho, (npoints, nfeatures, _sym_count(rho_dim))),
        dd_rho_sigma=_reshape(dd_rho_sigma, (npoints, nfeatures, rho_dim * sigma_dim)),
        dd_rho_tau=_reshape(dd_rho_tau, (npoints, nfeatures, rho_dim * tau_dim)),
        dd_sigma_sigma=_reshape(dd_sigma_sigma, (npoints, nfeatures, _sym_count(sigma_dim))),
        dd_sigma_tau=_reshape(dd_sigma_tau, (npoints, nfeatures, sigma_dim * tau_dim)),
        dd_tau_tau=_reshape(dd_tau_tau, (npoints, nfeatures, _sym_count(tau_dim))),
    )


def _as_rho_gradient(rho_gradient, npoints: int, *, spin: int) -> np.ndarray:
    arr = np.asarray(rho_gradient, dtype=np.float64, order="C")
    if spin == 0:
        if arr.shape != (3, npoints):
            raise ValueError(f"rho_gradient must have shape {(3, npoints)}, got {arr.shape}")
        return arr
    if spin == 1:
        if arr.shape == (2, 3, npoints):
            return np.asarray(arr.reshape(6, npoints), dtype=np.float64, order="C")
        if arr.shape == (6, npoints):
            return arr
        raise ValueError(f"rho_gradient must have shape {(2, 3, npoints)} or {(6, npoints)}, got {arr.shape}")
    raise ValueError(f"Unsupported spin {spin}")


def _eval_weighted_vvars(
    handle: ctypes.c_void_p | int,
    nfeatures: int,
    rho,
    sigma,
    tau,
    rho_gradient,
    weights,
    feature_gradient,
    *,
    spin: int,
    out=None,
) -> np.ndarray:
    rho_dim, sigma_dim, tau_dim = _spin_dims(spin)
    rho_arr = np.asarray(rho, dtype=np.float64, order="C")
    if rho_arr.ndim != 2 or rho_arr.shape[0] != rho_dim:
        raise ValueError(f"rho must have shape ({rho_dim}, npoints), got {rho_arr.shape}")
    npoints = int(rho_arr.shape[1])
    sigma_arr = _as_input(sigma, sigma_dim, npoints, "sigma")
    tau_arr = _as_input(tau, tau_dim, npoints, "tau")
    grad_arr = _as_rho_gradient(rho_gradient, npoints, spin=spin)
    weights_arr = _as_weights(weights, npoints)

    coeff = np.asarray(feature_gradient, dtype=np.float64, order="C").reshape(-1)
    nfeatures = int(nfeatures)
    if coeff.shape != (nfeatures,):
        raise ValueError(f"feature_gradient must have shape {(nfeatures,)}, got {coeff.shape}")

    out_shape = (5, npoints) if spin == 0 else (2, 5, npoints)
    weighted_vvars = _as_output(out, out_shape, "out")

    _sync_num_threads(handle)
    errbuf = ctypes.create_string_buffer(512)
    status = _lib.IFDFT_ifxc_evaluator_eval_weighted_vvars(
        ctypes.c_void_p(_handle_value(handle)),
        int(spin),
        ctypes.c_size_t(npoints),
        _ptr(rho_arr),
        _ptr(sigma_arr),
        _ptr(tau_arr),
        _ptr(grad_arr),
        _ptr(weights_arr),
        _ptr(coeff),
        _ptr(weighted_vvars),
        errbuf,
        ctypes.c_size_t(len(errbuf)),
    )
    _raise_on_status(status, errbuf)
    return weighted_vvars


def _as_pyscf_mgga_rho(rho, name: str) -> np.ndarray:
    arr = np.asarray(rho, dtype=np.float64, order="C")
    if arr.ndim != 2 or arr.shape[0] not in (5, 6):
        raise ValueError(f"{name} must have shape (5, npoints) or (6, npoints), got {arr.shape}")
    return arr


def _as_pyscf_mgga_pair(rho, *, spin: int, name: str) -> tuple[np.ndarray, np.ndarray | None]:
    if spin == 0:
        return _as_pyscf_mgga_rho(rho, name), None
    if spin == 1:
        if isinstance(rho, tuple) and len(rho) == 2:
            rho_a, rho_b = rho
        else:
            arr = np.asarray(rho, dtype=np.float64, order="C")
            if arr.ndim != 3 or arr.shape[0] != 2:
                raise ValueError(f"{name} must be an alpha/beta MGGA pair, got {arr.shape}")
            rho_a, rho_b = arr[0], arr[1]
        rho_a = _as_pyscf_mgga_rho(rho_a, f"{name}_a")
        rho_b = _as_pyscf_mgga_rho(rho_b, f"{name}_b")
        if rho_a.shape != rho_b.shape:
            raise ValueError(
                f"{name}_a and {name}_b must have the same shape, got {rho_a.shape} and {rho_b.shape}"
            )
        return rho_a, rho_b
    raise ValueError(f"Unsupported spin {spin}")


def _as_pyscf_mgga_rho_batch(rho, name: str) -> np.ndarray:
    arr = np.asarray(rho, dtype=np.float64, order="C")
    if arr.ndim != 3 or arr.shape[1] not in (5, 6):
        raise ValueError(f"{name} must have shape (ntrial, 5/6, npoints), got {arr.shape}")
    return arr


def _as_pyscf_mgga_pair_batch(rho, *, spin: int, name: str) -> tuple[np.ndarray, np.ndarray | None]:
    if spin == 0:
        return _as_pyscf_mgga_rho_batch(rho, name), None
    if spin == 1:
        if isinstance(rho, tuple) and len(rho) == 2:
            rho_a, rho_b = rho
        else:
            arr = np.asarray(rho, dtype=np.float64, order="C")
            if arr.ndim == 4 and arr.shape[1] == 2:
                rho_a, rho_b = arr[:, 0], arr[:, 1]
            elif arr.ndim == 4 and arr.shape[0] == 2:
                rho_a, rho_b = arr[0], arr[1]
            else:
                raise ValueError(f"{name} must be a batched alpha/beta MGGA pair, got {arr.shape}")
        rho_a = _as_pyscf_mgga_rho_batch(rho_a, f"{name}_a")
        rho_b = _as_pyscf_mgga_rho_batch(rho_b, f"{name}_b")
        if rho_a.shape != rho_b.shape:
            raise ValueError(
                f"{name}_a and {name}_b must have the same shape, got {rho_a.shape} and {rho_b.shape}"
            )
        return rho_a, rho_b
    raise ValueError(f"Unsupported spin {spin}")


def _require_matching_pyscf_mgga_shape(
    ref_a: np.ndarray,
    ref_b: np.ndarray | None,
    trial_a: np.ndarray,
    trial_b: np.ndarray | None,
    *,
    ref_name: str,
    trial_name: str,
) -> None:
    if ref_a.shape != trial_a.shape:
        raise ValueError(
            f"{ref_name}_a and {trial_name}_a must have the same shape, got {ref_a.shape} and {trial_a.shape}"
        )
    if (ref_b is None) != (trial_b is None):
        raise ValueError(f"{ref_name} and {trial_name} must have the same spin structure")
    if ref_b is not None and trial_b is not None and ref_b.shape != trial_b.shape:
        raise ValueError(
            f"{ref_name}_b and {trial_name}_b must have the same shape, got {ref_b.shape} and {trial_b.shape}"
        )


def _require_matching_pyscf_mgga_batch_shape(
    ref_a: np.ndarray,
    ref_b: np.ndarray | None,
    trial_a: np.ndarray,
    trial_b: np.ndarray | None,
    *,
    ref_name: str,
    trial_name: str,
) -> None:
    if ref_a.shape != trial_a.shape[1:]:
        raise ValueError(
            f"{ref_name}_a and {trial_name}_a batches must have matching MGGA shapes, "
            f"got {ref_a.shape} and {trial_a.shape}"
        )
    if (ref_b is None) != (trial_b is None):
        raise ValueError(f"{ref_name} and {trial_name} must have the same spin structure")
    if ref_b is not None and trial_b is not None and ref_b.shape != trial_b.shape[1:]:
        raise ValueError(
            f"{ref_name}_b and {trial_name}_b batches must have matching MGGA shapes, "
            f"got {ref_b.shape} and {trial_b.shape}"
        )


def _eval_weighted_vvars_from_pyscf_mgga(
    handle: ctypes.c_void_p | int,
    nfeatures: int,
    rho,
    weights,
    feature_gradient,
    *,
    spin: int,
    out=None,
) -> np.ndarray:
    rho_a, rho_b = _as_pyscf_mgga_pair(rho, spin=spin, name="rho")
    npoints = int(rho_a.shape[1])
    weights_arr = _as_weights(weights, npoints)
    coeff = np.asarray(feature_gradient, dtype=np.float64, order="C").reshape(-1)
    nfeatures = int(nfeatures)
    if coeff.shape != (nfeatures,):
        raise ValueError(f"feature_gradient must have shape {(nfeatures,)}, got {coeff.shape}")

    out_shape = (5, npoints) if spin == 0 else (2, 5, npoints)
    weighted_vvars = _as_output(out, out_shape, "out")
    _sync_num_threads(handle)
    errbuf = ctypes.create_string_buffer(512)
    status = _lib.IFDFT_ifxc_evaluator_eval_weighted_vvars_from_pyscf_mgga(
        ctypes.c_void_p(_handle_value(handle)),
        int(spin),
        ctypes.c_size_t(npoints),
        _ptr(rho_a),
        _ptr(rho_b),
        ctypes.c_int(int(rho_a.shape[0])),
        _ptr(weights_arr),
        _ptr(coeff),
        _ptr(weighted_vvars),
        errbuf,
        ctypes.c_size_t(len(errbuf)),
    )
    _raise_on_status(status, errbuf)
    return weighted_vvars


def _eval_weighted_vvars_batch_from_pyscf_mgga(
    handle: ctypes.c_void_p | int,
    nfeatures: int,
    rho,
    weights,
    feature_gradients,
    *,
    spin: int,
    out=None,
) -> np.ndarray:
    rho_a, rho_b = _as_pyscf_mgga_pair(rho, spin=spin, name="rho")
    npoints = int(rho_a.shape[1])
    weights_arr = _as_weights(weights, npoints)
    coeffs = np.asarray(feature_gradients, dtype=np.float64, order="C")
    nfeatures = int(nfeatures)
    if coeffs.ndim == 1:
        coeffs = coeffs.reshape(1, -1)
    if coeffs.ndim != 2 or coeffs.shape[1] != nfeatures:
        raise ValueError(f"feature_gradients must have shape (ncoeff, {nfeatures}), got {coeffs.shape}")
    ncoeff = int(coeffs.shape[0])

    out_shape = (ncoeff, 5, npoints) if spin == 0 else (ncoeff, 2, 5, npoints)
    weighted_vvars = _as_output(out, out_shape, "out")
    _sync_num_threads(handle)
    errbuf = ctypes.create_string_buffer(512)
    status = _lib.IFDFT_ifxc_evaluator_eval_weighted_vvars_batch_from_pyscf_mgga(
        ctypes.c_void_p(_handle_value(handle)),
        int(spin),
        ctypes.c_size_t(npoints),
        ctypes.c_size_t(ncoeff),
        _ptr(rho_a),
        _ptr(rho_b),
        ctypes.c_int(int(rho_a.shape[0])),
        _ptr(weights_arr),
        _ptr(coeffs),
        _ptr(weighted_vvars),
        errbuf,
        ctypes.c_size_t(len(errbuf)),
    )
    _raise_on_status(status, errbuf)
    return weighted_vvars


def _eval_features_from_pyscf_mgga(
    handle: ctypes.c_void_p | int,
    nfeatures: int,
    rho,
    weights,
    *,
    spin: int,
    out=None,
) -> np.ndarray:
    rho_a, rho_b = _as_pyscf_mgga_pair(rho, spin=spin, name="rho")
    npoints = int(rho_a.shape[1])
    weights_arr = _as_weights(weights, npoints)
    features = _as_output(out, (int(nfeatures),), "out")

    _sync_num_threads(handle)
    errbuf = ctypes.create_string_buffer(512)
    status = _lib.IFDFT_ifxc_evaluator_eval_features_from_pyscf_mgga(
        ctypes.c_void_p(_handle_value(handle)),
        int(spin),
        ctypes.c_size_t(npoints),
        _ptr(rho_a),
        _ptr(rho_b),
        ctypes.c_int(int(rho_a.shape[0])),
        _ptr(weights_arr),
        _ptr(features),
        errbuf,
        ctypes.c_size_t(len(errbuf)),
    )
    _raise_on_status(status, errbuf)
    return features


def _eval_feature_directional_from_pyscf_mgga(
    handle: ctypes.c_void_p | int,
    nfeatures: int,
    rho0,
    rho1,
    weights,
    *,
    spin: int,
    out=None,
) -> np.ndarray:
    rho0_a, rho0_b = _as_pyscf_mgga_pair(rho0, spin=spin, name="rho0")
    rho1_a, rho1_b = _as_pyscf_mgga_pair(rho1, spin=spin, name="rho1")
    _require_matching_pyscf_mgga_shape(
        rho0_a, rho0_b, rho1_a, rho1_b, ref_name="rho0", trial_name="rho1"
    )
    npoints = int(rho0_a.shape[1])
    weights_arr = _as_weights(weights, npoints)
    dfeatures = _as_output(out, (int(nfeatures),), "out")

    _sync_num_threads(handle)
    errbuf = ctypes.create_string_buffer(512)
    status = _lib.IFDFT_ifxc_evaluator_eval_feature_directional_from_pyscf_mgga(
        ctypes.c_void_p(_handle_value(handle)),
        int(spin),
        ctypes.c_size_t(npoints),
        _ptr(rho0_a),
        _ptr(rho0_b),
        _ptr(rho1_a),
        _ptr(rho1_b),
        ctypes.c_int(int(rho0_a.shape[0])),
        _ptr(weights_arr),
        _ptr(dfeatures),
        errbuf,
        ctypes.c_size_t(len(errbuf)),
    )
    _raise_on_status(status, errbuf)
    return dfeatures


def _eval_weighted_response_from_pyscf_mgga(
    handle: ctypes.c_void_p | int,
    nfeatures: int,
    rho0,
    rho1,
    weights,
    feature_gradient,
    *,
    spin: int,
    out=None,
) -> np.ndarray:
    rho0_a, rho0_b = _as_pyscf_mgga_pair(rho0, spin=spin, name="rho0")
    rho1_a, rho1_b = _as_pyscf_mgga_pair(rho1, spin=spin, name="rho1")
    _require_matching_pyscf_mgga_shape(
        rho0_a, rho0_b, rho1_a, rho1_b, ref_name="rho0", trial_name="rho1"
    )
    npoints = int(rho0_a.shape[1])
    weights_arr = _as_weights(weights, npoints)
    coeff = np.asarray(feature_gradient, dtype=np.float64, order="C").reshape(-1)
    nfeatures = int(nfeatures)
    if coeff.shape != (nfeatures,):
        raise ValueError(f"feature_gradient must have shape {(nfeatures,)}, got {coeff.shape}")

    out_shape = (5, npoints) if spin == 0 else (2, 5, npoints)
    weighted_vvars = _as_output(out, out_shape, "out")
    _sync_num_threads(handle)
    errbuf = ctypes.create_string_buffer(512)
    status = _lib.IFDFT_ifxc_evaluator_eval_weighted_response_from_pyscf_mgga(
        ctypes.c_void_p(_handle_value(handle)),
        int(spin),
        ctypes.c_size_t(npoints),
        _ptr(rho0_a),
        _ptr(rho0_b),
        _ptr(rho1_a),
        _ptr(rho1_b),
        ctypes.c_int(int(rho0_a.shape[0])),
        _ptr(weights_arr),
        _ptr(coeff),
        _ptr(weighted_vvars),
        errbuf,
        ctypes.c_size_t(len(errbuf)),
    )
    _raise_on_status(status, errbuf)
    return weighted_vvars


def _eval_response_batch_from_pyscf_mgga(
    handle: ctypes.c_void_p | int,
    nfeatures: int,
    rho0,
    rho1_batch,
    weights,
    feature_gradient,
    *,
    spin: int,
    out_kernel=None,
    out_dfeatures=None,
) -> tuple[np.ndarray, np.ndarray]:
    rho0_a, rho0_b = _as_pyscf_mgga_pair(rho0, spin=spin, name="rho0")
    rho1_a, rho1_b = _as_pyscf_mgga_pair_batch(rho1_batch, spin=spin, name="rho1_batch")
    _require_matching_pyscf_mgga_batch_shape(
        rho0_a, rho0_b, rho1_a, rho1_b, ref_name="rho0", trial_name="rho1_batch"
    )
    npoints = int(rho0_a.shape[1])
    ntrial = int(rho1_a.shape[0])
    weights_arr = _as_weights(weights, npoints)
    coeff = np.asarray(feature_gradient, dtype=np.float64, order="C").reshape(-1)
    nfeatures = int(nfeatures)
    if coeff.shape != (nfeatures,):
        raise ValueError(f"feature_gradient must have shape {(nfeatures,)}, got {coeff.shape}")

    kernel_shape = (ntrial, 5, npoints) if spin == 0 else (ntrial, 2, 5, npoints)
    weighted_kernel = _as_output(out_kernel, kernel_shape, "out_kernel")
    dfeatures = _as_output(out_dfeatures, (ntrial, nfeatures), "out_dfeatures")
    _sync_num_threads(handle)
    errbuf = ctypes.create_string_buffer(512)
    status = _lib.IFDFT_ifxc_evaluator_eval_response_batch_from_pyscf_mgga(
        ctypes.c_void_p(_handle_value(handle)),
        int(spin),
        ctypes.c_size_t(npoints),
        ctypes.c_size_t(ntrial),
        _ptr(rho0_a),
        _ptr(rho0_b),
        _ptr(rho1_a),
        _ptr(rho1_b),
        ctypes.c_int(int(rho0_a.shape[0])),
        _ptr(weights_arr),
        _ptr(coeff),
        _ptr(weighted_kernel),
        _ptr(dfeatures),
        errbuf,
        ctypes.c_size_t(len(errbuf)),
    )
    _raise_on_status(status, errbuf)
    return weighted_kernel, dfeatures


def _eval_feature_directional_derivative(
    handle: ctypes.c_void_p | int,
    nfeatures: int,
    rho,
    sigma,
    tau,
    drho,
    dsigma,
    dtau,
    weights,
    *,
    spin: int,
    out=None,
) -> np.ndarray:
    rho_dim, sigma_dim, tau_dim = _spin_dims(spin)
    rho_arr = np.asarray(rho, dtype=np.float64, order="C")
    if rho_arr.ndim != 2 or rho_arr.shape[0] != rho_dim:
        raise ValueError(f"rho must have shape ({rho_dim}, npoints), got {rho_arr.shape}")
    npoints = int(rho_arr.shape[1])
    sigma_arr = _as_input(sigma, sigma_dim, npoints, "sigma")
    tau_arr = _as_input(tau, tau_dim, npoints, "tau")
    drho_arr = _as_input(drho, rho_dim, npoints, "drho")
    dsigma_arr = _as_input(dsigma, sigma_dim, npoints, "dsigma")
    dtau_arr = _as_input(dtau, tau_dim, npoints, "dtau")
    weights_arr = _as_weights(weights, npoints)

    dfeatures = _as_output(out, (int(nfeatures),), "out")
    _sync_num_threads(handle)
    errbuf = ctypes.create_string_buffer(512)
    status = _lib.IFDFT_ifxc_evaluator_eval_feature_directional_derivative(
        ctypes.c_void_p(_handle_value(handle)),
        int(spin),
        ctypes.c_size_t(npoints),
        _ptr(rho_arr),
        _ptr(sigma_arr),
        _ptr(tau_arr),
        _ptr(drho_arr),
        _ptr(dsigma_arr),
        _ptr(dtau_arr),
        _ptr(weights_arr),
        _ptr(dfeatures),
        errbuf,
        ctypes.c_size_t(len(errbuf)),
    )
    _raise_on_status(status, errbuf)
    return dfeatures


def _eval_weighted_response_vvars(
    handle: ctypes.c_void_p | int,
    nfeatures: int,
    rho,
    sigma,
    tau,
    drho,
    dsigma,
    dtau,
    rho_gradient,
    drho_gradient,
    weights,
    feature_gradient,
    *,
    spin: int,
    out=None,
) -> np.ndarray:
    rho_dim, sigma_dim, tau_dim = _spin_dims(spin)
    rho_arr = np.asarray(rho, dtype=np.float64, order="C")
    if rho_arr.ndim != 2 or rho_arr.shape[0] != rho_dim:
        raise ValueError(f"rho must have shape ({rho_dim}, npoints), got {rho_arr.shape}")
    npoints = int(rho_arr.shape[1])
    sigma_arr = _as_input(sigma, sigma_dim, npoints, "sigma")
    tau_arr = _as_input(tau, tau_dim, npoints, "tau")
    drho_arr = _as_input(drho, rho_dim, npoints, "drho")
    dsigma_arr = _as_input(dsigma, sigma_dim, npoints, "dsigma")
    dtau_arr = _as_input(dtau, tau_dim, npoints, "dtau")
    grad_arr = _as_rho_gradient(rho_gradient, npoints, spin=spin)
    dgrad_arr = _as_rho_gradient(drho_gradient, npoints, spin=spin)
    weights_arr = _as_weights(weights, npoints)

    coeff = np.asarray(feature_gradient, dtype=np.float64, order="C").reshape(-1)
    nfeatures = int(nfeatures)
    if coeff.shape != (nfeatures,):
        raise ValueError(f"feature_gradient must have shape {(nfeatures,)}, got {coeff.shape}")

    out_shape = (5, npoints) if spin == 0 else (2, 5, npoints)
    weighted_vvars = _as_output(out, out_shape, "out")
    _sync_num_threads(handle)
    errbuf = ctypes.create_string_buffer(512)
    status = _lib.IFDFT_ifxc_evaluator_eval_weighted_response_vvars(
        ctypes.c_void_p(_handle_value(handle)),
        int(spin),
        ctypes.c_size_t(npoints),
        _ptr(rho_arr),
        _ptr(sigma_arr),
        _ptr(tau_arr),
        _ptr(drho_arr),
        _ptr(dsigma_arr),
        _ptr(dtau_arr),
        _ptr(grad_arr),
        _ptr(dgrad_arr),
        _ptr(weights_arr),
        _ptr(coeff),
        _ptr(weighted_vvars),
        errbuf,
        ctypes.c_size_t(len(errbuf)),
    )
    _raise_on_status(status, errbuf)
    return weighted_vvars
