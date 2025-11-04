"""Visdom integration callbacks for Ultralytics training and validation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING, colorstr

try:  # Verify integration is enabled
    assert not TESTS_RUNNING
    assert SETTINGS.get("visdom", False) is True
    import numpy as np
    from visdom import Visdom

    VISDOM_ENABLED = True
except (ImportError, AssertionError, AttributeError, KeyError):
    Visdom = None
    np = None
    VISDOM_ENABLED = False

PREFIX = colorstr("Visdom: ")
_ENV = str(SETTINGS.get("visdom_env", "ultralytics"))
_SERVER = str(SETTINGS.get("visdom_server", "http://localhost"))
_PORT = int(SETTINGS.get("visdom_port", 8097))
_USERNAME = SETTINGS.get("visdom_username")
_PASSWORD = SETTINGS.get("visdom_password")

_CLIENT: Visdom | None = None
_CLIENT_FAILED = False
_SCALAR_WINDOWS: dict[str, str] = {}
_IMAGE_WINDOWS: dict[str, str] = {}


def _ensure_client() -> Visdom | None:
    """Create a Visdom client instance if possible."""
    global _CLIENT, _CLIENT_FAILED
    if not VISDOM_ENABLED or _CLIENT_FAILED:
        return None
    if _CLIENT is None:
        try:
            kwargs: dict[str, Any] = {"server": _SERVER, "port": _PORT, "env": _ENV, "use_incoming_socket": False}
            if _USERNAME:
                kwargs["username"] = _USERNAME
            if _PASSWORD:
                kwargs["password"] = _PASSWORD
            _CLIENT = Visdom(**kwargs)
            if not _CLIENT.check_connection():
                raise ConnectionError("unable to connect to Visdom server")
            LOGGER.info(f"{PREFIX}Connected to {_SERVER}:{_PORT} (env='{_ENV}')")
        except Exception as err:
            LOGGER.warning(f"{PREFIX}Visdom client disabled: {err}")
            _CLIENT = None
            _CLIENT_FAILED = True
    return _CLIENT


def _log_scalars(values: dict[str, float | int | Any], step: float) -> None:
    client = _ensure_client()
    if client is None or not values:
        return
    for key, value in values.items():
        try:
            scalar = float(value)
            if math.isnan(scalar) or math.isinf(scalar):
                continue
        except (TypeError, ValueError):
            continue
        x = np.array([step], dtype=np.float32)
        y = np.array([scalar], dtype=np.float32)
        window = _SCALAR_WINDOWS.get(key)
        opts = {"title": key, "xlabel": "epoch", "ylabel": key.split("/")[-1]}
        if window is None:
            _SCALAR_WINDOWS[key] = client.line(X=x, Y=y, opts=opts)
        else:
            client.line(X=x, Y=y, win=window, update="append")


def _prepare_image(array: np.ndarray) -> np.ndarray | None:
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3:
        return None
    if array.shape[2] == 4:
        array = array[..., :3]
    if array.shape[0] <= 4 and array.shape[1] > 4 and array.shape[2] > 4:
        chw = array.astype(np.float32)
    else:
        chw = np.transpose(array, (2, 0, 1)).astype(np.float32)
    max_value = float(chw.max()) if chw.size else 0.0
    if max_value > 1.0:
        chw /= 255.0
    return np.clip(chw, 0.0, 1.0)


def _flush_plots(owner, scope: str) -> None:
    client = _ensure_client()
    if client is None:
        return
    plots = getattr(owner, "plots", None)
    if not plots:
        return
    import cv2  # scoped import to avoid mandatory dependency when Visdom disabled

    for path, payload in list(plots.items()):
        data = payload.get("data") if isinstance(payload, dict) else None
        if data is not None and np is not None:
            array = np.asarray(data)
        else:
            img_path = Path(path)
            if not img_path.exists():
                plots.pop(path, None)
                continue
            image = cv2.imread(str(img_path))
            if image is None:
                plots.pop(path, None)
                continue
            array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed = _prepare_image(array)
        if processed is None:
            plots.pop(path, None)
            continue
        title = f"{scope}/{Path(path).stem}"
        window = _IMAGE_WINDOWS.get(title)
        opts = {"title": title}
        if window is None:
            _IMAGE_WINDOWS[title] = client.image(processed, opts=opts)
        else:
            client.image(processed, win=window, opts=opts)
        plots.pop(path, None)


def on_pretrain_routine_start(trainer) -> None:
    """Instantiate the Visdom client before training starts."""
    _ensure_client()


def on_train_batch_end(trainer) -> None:
    """Send any queued training plots to Visdom."""
    _flush_plots(trainer, "train")


def on_train_epoch_end(trainer) -> None:
    """Log training loss components and learning rates."""
    if trainer.tloss is not None:
        losses = trainer.label_loss_items(trainer.tloss, prefix="train")
        _log_scalars(losses, trainer.epoch + 1)
    if getattr(trainer, "lr", None):
        _log_scalars(trainer.lr, trainer.epoch + 1)


def on_fit_epoch_end(trainer) -> None:
    """Log validation metrics and flush pending plots after each epoch."""
    if getattr(trainer, "metrics", None):
        _log_scalars(trainer.metrics, trainer.epoch + 1)
    _flush_plots(trainer, "train")


def on_val_end(validator) -> None:
    """Handle validation plots and metrics for Visdom."""
    if not getattr(validator, "training", False):
        metrics = getattr(validator, "metrics", None)
        if metrics is not None and hasattr(metrics, "results_dict"):
            step = getattr(validator, "_visdom_step", 0) + 1
            validator._visdom_step = step
            _log_scalars(metrics.results_dict, step)
    _flush_plots(validator, "val")


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_batch_end": on_train_batch_end,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_val_end": on_val_end,
    }
    if VISDOM_ENABLED
    else {}
)
