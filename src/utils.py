import numpy as np
import os
import base64
import tempfile
import tensorflow as tf

PURPOSES = ["근무", "정비", "복지", "기타"]
DESTINATIONS = ["본부", "병영", "창고", "식당"]
TIMES = ["주간", "야간"]

PURPOSE_MAP = {p: i for i, p in enumerate(PURPOSES)}
DESTINATION_MAP = {d: i for i, d in enumerate(DESTINATIONS)}
TIME_MAP = {t: i for i, t in enumerate(TIMES)}

INPUT_SIZE = 14
LATENT_SIZE = 4


def encode_info(purpose: str, dest: str, time: str) -> np.ndarray:
    """Encode text info into one-hot numpy array of length 10."""
    p_vec = np.zeros(len(PURPOSES))
    d_vec = np.zeros(len(DESTINATIONS))
    t_vec = np.zeros(len(TIMES))
    p_vec[PURPOSE_MAP[purpose]] = 1
    d_vec[DESTINATION_MAP[dest]] = 1
    t_vec[TIME_MAP[time]] = 1
    return np.concatenate([p_vec, d_vec, t_vec])


def save_model_b64(model: tf.keras.Model, path: str) -> None:
    """Save a Keras model to a base64-encoded text file."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        model.save(tmp.name)
        tmp.seek(0)
        data = tmp.read()
    os.remove(tmp.name)
    b64 = base64.b64encode(data).decode("utf-8")
    with open(path, "w", encoding="utf-8") as f:
        f.write(b64)


def load_model_b64(path: str) -> tf.keras.Model:
    """Load a Keras model from a base64-encoded text file."""
    with open(path, "r", encoding="utf-8") as f:
        b64 = f.read()
    data = base64.b64decode(b64.encode("utf-8"))
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        model = tf.keras.models.load_model(tmp.name, compile=False)
    os.remove(tmp.name)
    return model


