import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

keras = None
np = None
sf = None
tf = None
hub = None
resample_poly = None
BertTokenizer = None
TFBertModel = None


QUESTION = "Is this audio real human speech or spoofed synthetic speech?"


def load_training_dependencies():
    global keras, np, sf, tf, hub, resample_poly, BertTokenizer, TFBertModel
    if keras is not None:
        return

    import keras as keras_module
    import numpy as np_module
    import soundfile as sf_module
    import tensorflow as tf_module
    import tensorflow_hub as hub_module
    from scipy.signal import resample_poly as scipy_resample_poly
    from transformers import BertTokenizer as TransformersBertTokenizer
    from transformers import TFBertModel as TransformersTFBertModel

    keras = keras_module
    np = np_module
    sf = sf_module
    tf = tf_module
    hub = hub_module
    resample_poly = scipy_resample_poly
    BertTokenizer = TransformersBertTokenizer
    TFBertModel = TransformersTFBertModel


def load_protocol(protocol_path):
    rows = []
    with open(protocol_path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=" ")
        for row in reader:
            row = [part for part in row if part]
            if len(row) < 5:
                continue
            rows.append({"file_id": row[1], "label": 0.0 if row[4] == "bonafide" else 1.0})
    return rows


def read_audio_16k_mono(path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        gcd = np.gcd(sample_rate, 16000)
        audio = resample_poly(audio, 16000 // gcd, sample_rate // gcd).astype("float32")
    return audio


def extract_audio_embeddings(rows, audio_dir, cache_path, yamnet):
    cache_path = Path(cache_path)
    if cache_path.exists():
        data = np.load(cache_path)
        if data.shape == (len(rows), 1024):
            return data.astype("float32")

    embeddings = np.zeros((len(rows), 1024), dtype="float32")
    for index, row in enumerate(rows, start=1):
        audio_path = Path(audio_dir) / f"{row['file_id']}.flac"
        audio = read_audio_16k_mono(audio_path)
        _, frame_embeddings, _ = yamnet(audio)
        embeddings[index - 1] = tf.reduce_mean(frame_embeddings, axis=0).numpy()
        if index % 250 == 0:
            print(f"  extracted {index:,}/{len(rows):,} audio embeddings")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


def extract_question_embedding():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = TFBertModel.from_pretrained("bert-base-uncased", use_safetensors=False)
    tokens = tokenizer(QUESTION, return_tensors="tf", truncation=True, padding=True, max_length=32)
    return bert(**tokens).pooler_output[0].numpy().astype("float32")


def build_model(audio_dim=1024, text_dim=768, dropout=0.5, l2=1e-4):
    regularizer = keras.regularizers.l2(l2)

    audio_input = keras.layers.Input(shape=(audio_dim,), name="audio_input")
    text_input = keras.layers.Input(shape=(text_dim,), name="text_input")

    audio = keras.layers.GaussianNoise(0.03)(audio_input)
    audio = keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizer)(audio)
    audio = keras.layers.BatchNormalization()(audio)
    audio = keras.layers.Dropout(dropout)(audio)

    text = keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer)(text_input)
    text = keras.layers.BatchNormalization()(text)
    text = keras.layers.Dropout(dropout)(text)

    fusion = keras.layers.Concatenate()([audio, text])
    fusion = keras.layers.Dense(96, activation="relu", kernel_regularizer=regularizer)(fusion)
    fusion = keras.layers.BatchNormalization()(fusion)
    fusion = keras.layers.Dropout(dropout)(fusion)

    output = keras.layers.Dense(1, activation="sigmoid", name="spoof_probability")(fusion)
    model = keras.Model(inputs={"audio_input": audio_input, "text_input": text_input}, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-5),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def make_dataset(x_audio, x_text, y, batch_size, training):
    dataset = tf.data.Dataset.from_tensor_slices(({"audio_input": x_audio, "text_input": x_text}, y))
    if training:
        dataset = dataset.shuffle(min(len(y), 10000), reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def class_weights(y):
    labels, counts = np.unique(y, return_counts=True)
    total = len(y)
    return {float(label): total / (len(labels) * count) for label, count in zip(labels, counts)}


def main():
    parser = argparse.ArgumentParser(description="Retrain the YAMNet+BERT fusion model with anti-overfitting controls.")
    parser.add_argument("--la-root", default="../LA", help="Path containing ASVspoof2019_LA_* folders")
    parser.add_argument("--output", default="fusion_model_fitted.keras")
    parser.add_argument("--cache-dir", default="training_cache")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=1e-4)
    args = parser.parse_args()

    la_root = Path(args.la_root)
    protocol_dir = la_root / "ASVspoof2019_LA_cm_protocols"
    train_protocol = protocol_dir / "ASVspoof2019.LA.cm.train.trn.txt"
    dev_protocol = protocol_dir / "ASVspoof2019.LA.cm.dev.trl.txt"
    if not train_protocol.exists() or not dev_protocol.exists():
        raise FileNotFoundError(
            "ASVspoof2019 LA protocol files were not found. "
            f"Expected {train_protocol} and {dev_protocol}."
        )

    load_training_dependencies()

    train_rows = load_protocol(train_protocol)
    dev_rows = load_protocol(dev_protocol)

    overlap = {row["file_id"] for row in train_rows}.intersection(row["file_id"] for row in dev_rows)
    if overlap:
        raise RuntimeError(f"Train/dev leakage detected: {len(overlap)} overlapping file ids")

    print("Loading feature extractors...")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    question_embedding = extract_question_embedding()

    print("Extracting or loading cached YAMNet embeddings...")
    x_audio_train = extract_audio_embeddings(
        train_rows,
        la_root / "ASVspoof2019_LA_train" / "flac",
        Path(args.cache_dir) / "x_audio_train.npy",
        yamnet,
    )
    x_audio_dev = extract_audio_embeddings(
        dev_rows,
        la_root / "ASVspoof2019_LA_dev" / "flac",
        Path(args.cache_dir) / "x_audio_dev.npy",
        yamnet,
    )

    y_train = np.array([row["label"] for row in train_rows], dtype="float32")
    y_dev = np.array([row["label"] for row in dev_rows], dtype="float32")
    x_text_train = np.tile(question_embedding, (len(train_rows), 1)).astype("float32")
    x_text_dev = np.tile(question_embedding, (len(dev_rows), 1)).astype("float32")

    model = build_model(dropout=args.dropout, l2=args.l2)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=3, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(args.output, monitor="val_auc", mode="max", save_best_only=True),
        keras.callbacks.CSVLogger("training_history.csv"),
    ]

    history = model.fit(
        make_dataset(x_audio_train, x_text_train, y_train, args.batch_size, training=True),
        validation_data=make_dataset(x_audio_dev, x_text_dev, y_dev, args.batch_size, training=False),
        epochs=args.epochs,
        class_weight=class_weights(y_train),
        callbacks=callbacks,
    )

    best_val_auc = max(history.history.get("val_auc", [float("nan")]))
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Saved best model to {args.output}")


if __name__ == "__main__":
    main()
