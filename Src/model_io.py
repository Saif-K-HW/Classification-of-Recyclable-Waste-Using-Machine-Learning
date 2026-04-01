"""
Model I/O Helpers
Centralized safe model loading for trusted local checkpoints
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

# Importing model registers custom serializable objects (for example BackbonePreprocess)
# so checkpoints can be deserialized in prediction/dashboard contexts.
import model  # noqa: F401


def load_saved_model(model_path: str | Path, compile_model: bool = False) -> tf.keras.Model:
    # Experiment checkpoints are locally generated, so disabling Keras safe_mode is expected here.
    kwargs = {"compile": compile_model}
    try:
        return tf.keras.models.load_model(model_path, safe_mode=False, **kwargs)
    except TypeError:
        # Older TF/Keras builds may not expose safe_mode in load_model.
        return tf.keras.models.load_model(model_path, **kwargs)
