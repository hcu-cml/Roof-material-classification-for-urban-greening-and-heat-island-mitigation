"""Model fitting utilities for the green roof scenario."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

__all__ = [
    "sample_model_inputs",
    "fit_model",
    "predict_model",
    "predict_partial",
]


def _validate_same_shape(**arrays: np.ndarray) -> tuple[int, ...]:
    """Validate that all named model arrays have one common shape."""
    shapes = {name: arr.shape for name, arr in arrays.items()}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) != 1:
        details = ", ".join(f"{name}={shape}" for name, shape in shapes.items())
        raise ValueError(f"All model arrays must have the same shape; got {details}.")
    return next(iter(unique_shapes))


def sample_model_inputs(
    lst: np.ndarray,
    ndvi: np.ndarray,
    albedo: np.ndarray,
    ndbi: np.ndarray,
    frac: float,
    seed: int,
    block_size: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample finite predictor and LST pixels, optionally with spatial thinning."""
    _validate_same_shape(lst=lst, ndvi=ndvi, albedo=albedo, ndbi=ndbi)
    if not 0 < frac <= 1:
        raise ValueError("frac must be greater than 0 and at most 1.")
    mask = np.isfinite(lst) & np.isfinite(ndvi) & np.isfinite(albedo) & np.isfinite(ndbi)

    if block_size is None or block_size <= 1:
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            raise ValueError("No valid pixels for modeling.")
        rng = np.random.default_rng(seed)
        n = max(1000, int(frac * len(idx)))
        sel = rng.choice(idx, size=min(n, len(idx)), replace=False)
        X = np.column_stack([ndvi.flat[sel], albedo.flat[sel], ndbi.flat[sel]])
        y = lst.flat[sel]
        return X, y

    height, width = lst.shape
    rng = np.random.default_rng(seed)
    selected_idx: list[int] = []

    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            submask = mask[row : row + block_size, col : col + block_size]
            if not submask.any():
                continue
            ys, xs = np.where(submask)
            j = rng.integers(0, len(ys))
            r = row + ys[j]
            c = col + xs[j]
            selected_idx.append(r * width + c)

    if not selected_idx:
        raise ValueError("No valid pixels for modeling after spatial thinning.")

    idx = np.array(selected_idx, dtype=int)
    n = max(1000, int(frac * len(idx)))
    n = min(n, len(idx))
    sel = rng.choice(idx, size=n, replace=False)
    X = np.column_stack([ndvi.flat[sel], albedo.flat[sel], ndbi.flat[sel]])
    y = lst.flat[sel]
    return X, y


def fit_model(
    lst: np.ndarray,
    ndvi: np.ndarray,
    albedo: np.ndarray,
    ndbi: np.ndarray,
    *,
    frac: float = 0.1,
    seed: int = 42,
    model_type: str = "linear",
    block_size: int | None = None,
):
    """Fit a regression model and return it with train/test metrics."""
    if model_type not in {"linear", "rf"}:
        raise ValueError("model_type must be 'linear' or 'rf'.")
    X, y = sample_model_inputs(lst, ndvi, albedo, ndbi, frac, seed, block_size=block_size)
    if len(y) < 6:
        raise ValueError(
            "At least six valid training pixels are required to fit and evaluate a model."
        )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    if model_type == "linear":
        model = LinearRegression().fit(X_train, y_train)
    else:
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=12,          # Limit depth to prevent complex "if-then" chains
            min_samples_leaf=5,    # Require at least 5 pixels to make a decision
            max_features="sqrt",   # (Optional) Forces trees to use different variables
            random_state=seed,
            n_jobs=-1,
        ).fit(X_train, y_train)

    r2_train = model.score(X_train, y_train)
    rmse_train = float(np.sqrt(np.mean((y_train - model.predict(X_train)) ** 2)))
    r2_test = model.score(X_test, y_test)
    rmse_test = float(np.sqrt(np.mean((y_test - model.predict(X_test)) ** 2)))

    metrics = {
        "r2_train": r2_train,
        "rmse_train": rmse_train,
        "r2_test": r2_test,
        "rmse_test": rmse_test,
    }
    return model, metrics


def predict_model(model, ndvi: np.ndarray, albedo: np.ndarray, ndbi: np.ndarray) -> np.ndarray:
    """Predict a full raster while leaving non-finite inputs as NaN."""
    _validate_same_shape(ndvi=ndvi, albedo=albedo, ndbi=ndbi)
    mask = np.isfinite(ndvi) & np.isfinite(albedo) & np.isfinite(ndbi)
    out = np.full(ndvi.shape, np.nan, dtype="float32")
    if mask.sum() == 0:
        return out
    X = np.column_stack([ndvi.ravel()[mask.ravel()], albedo.ravel()[mask.ravel()], ndbi.ravel()[mask.ravel()]])
    preds = model.predict(X).astype("float32")
    out.ravel()[mask.ravel()] = preds
    return out


def predict_partial(
    model,
    ndvi: np.ndarray,
    albedo: np.ndarray,
    mask: np.ndarray,
    ndbi: np.ndarray,
) -> np.ndarray:
    """Predict only finite pixels enabled by a supplied boolean mask."""
    _validate_same_shape(ndvi=ndvi, albedo=albedo, ndbi=ndbi, mask=mask)
    out = np.full(ndvi.shape, np.nan, dtype="float32")
    finite_mask = mask & np.isfinite(ndvi) & np.isfinite(albedo) & np.isfinite(ndbi)
    idx = np.flatnonzero(finite_mask)
    if idx.size == 0:
        return out
    X = np.column_stack([ndvi.ravel()[idx], albedo.ravel()[idx], ndbi.ravel()[idx]])
    yhat = model.predict(X).astype("float32")
    out.ravel()[idx] = yhat
    return out
