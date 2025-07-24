import os
import sys
import random
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR   = PROJECT_ROOT / "data"
TEST_DIR   = PROJECT_ROOT / "test"
LABELS_TXT = PROJECT_ROOT / "labels.txt"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5
TRAIN_SPLIT = 0.8 
LABEL_SELECTION_MODE = "dominant"  # "dominant" / "first" / "avg"
SEED = 1337

#   LABEL_SELECTION_MODE:
#   “dominant”  -> first line with a single color (if none: fallback to “first”)
#   “first”      -> first line that occurs for a given file (regardless of the number of colors, takes the first color from the list)
#   “avg”        -> average all colors from all lines (all HEX codes) -> 1 LAB

def _parse_hex(hex_color: str):
    hex_color = hex_color.strip()
    if not hex_color.startswith('#'):
        raise ValueError(f"Color without #: {hex_color}")
    if len(hex_color) != 7:
        raise ValueError(f"Incorrect HEX format {hex_color}")
    return hex_color


def hex_to_lab01(hex_color: str) -> np.ndarray:
    hex_color = _parse_hex(hex_color)
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0]  # shape (3,), dtype uint8
    lab01 = lab.astype(np.float32) / 255.0
    return lab01  # np.float32 [L,a,b] w 0..1


def merge_labs_avg(labs_list):
    if len(labs_list) == 1:
        return labs_list[0]
    return np.mean(np.stack(labs_list, axis=0), axis=0).astype(np.float32)


def load_labels_file(labels_path: str):
    data = {}
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"No labels file in: {labels_path}")

    with open(labels_path, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            fname = parts[0].strip()
            colors_str = "".join(parts[1:])
            colors = [c.strip() for c in colors_str.split(',') if c.strip()]
            labs = []
            for c in colors:
                try:
                    labs.append(hex_to_lab01(c))
                except Exception as e:
                    print(f"[WARN] Pomijam kolor '{c}' dla pliku {fname}: {e}")
            if not labs:
                continue
            data.setdefault(fname, []).append(labs)
    return data  # dict[str, list[list[lab01]]]


def select_label_for_file(labs_lines_for_file, mode="dominant"):
    if mode == "dominant":
        for labs_line in labs_lines_for_file:
            if len(labs_line) == 1:
                return labs_line[0]
        # fallback
        mode = "first"

    if mode == "first":
        return labs_lines_for_file[0][0]

    if mode == "avg":
        all_labs = []
        for labs_line in labs_lines_for_file:
            all_labs.extend(labs_line)
        return merge_labs_avg(all_labs)

    raise ValueError(f"Unknown mode: {mode}")

def list_images_jpg(dir_path: str):
    print(dir_path)
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"No matching dir path: {dir_path}")
    files = [f for f in os.listdir(dir_path) if f.lower().endswith('.jpg')]
    files.sort()
    return files

def load_and_preprocess_image_tf(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def make_tf_dataset(image_paths, labels, batch_size=BATCH_SIZE, shuffle=True):
    paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels.astype(np.float32))
    ds = tf.data.Dataset.zip((paths_ds, labels_ds))
    def _map(path, label):
        img = load_and_preprocess_image_tf(path)
        return img, label
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# =============================
# MODEL
# =============================

def build_model(input_shape=(128,128,3)):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='sigmoid'),  # LAB w 0..1
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    print("TensorFlow:", tf.__version__)
    print("NumPy:", np.__version__)
    print(DATA_DIR)
    print(os.path.isdir("/Users/mateuszkolber/Desktop/projects/biai_2025/data/"))
    if np.lib.NumpyVersion(np.__version__) >= np.lib.NumpyVersion('2.0.0'):
        print("Change numpy to < 2.0.0")

    img_names = list_images_jpg(DATA_DIR)
    print(f"Found {len(img_names)} of photos in {DATA_DIR}.")

    labels_raw = load_labels_file(LABELS_TXT)
    print(f"Number of 'answers' provided to model: {len(labels_raw)}")

    image_paths = []
    image_labels = []
    missing_labels = []

    for fname in img_names:
        if fname not in labels_raw:
            missing_labels.append(fname)
            continue
        lab_vec = select_label_for_file(labels_raw[fname], mode=LABEL_SELECTION_MODE)
        image_paths.append(os.path.join(DATA_DIR, fname))
        image_labels.append(lab_vec)

    image_paths = np.array(image_paths, dtype=object)
    image_labels = np.stack(image_labels, axis=0).astype(np.float32)  # shape (N,3)

    print(f"Labeled photos: {len(image_paths)}")
    if missing_labels:
        print(f"Number of photos without answears: {len(missing_labels)}")

    n_total = len(image_paths)
    n_train = int(TRAIN_SPLIT * n_total)
    idxs = list(range(n_total))
    random.Random(SEED).shuffle(idxs)
    train_idxs = idxs[:n_train]
    val_idxs = idxs[n_train:]

    train_paths = image_paths[train_idxs]
    train_labels = image_labels[train_idxs]
    val_paths = image_paths[val_idxs]
    val_labels = image_labels[val_idxs]

    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")

    train_ds = make_tf_dataset(train_paths, train_labels, batch_size=BATCH_SIZE, shuffle=True)
    val_ds = make_tf_dataset(val_paths, val_labels, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model.summary()

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    out_path = os.path.join(PROJECT_ROOT, "lab_color_model.keras")
    model.save(out_path)
    print(f"Model saved in: {out_path}")

    if len(val_paths) > 0:
        sample_path = val_paths[0]
        sample_img = load_and_preprocess_image_tf(sample_path)
        sample_img = tf.expand_dims(sample_img, axis=0)  # batch=1
        pred_lab01 = model.predict(sample_img)[0]
        print(f"Example prediction {sample_path} (LAB 0..1):", pred_lab01)


if __name__ == "__main__":
    main()
