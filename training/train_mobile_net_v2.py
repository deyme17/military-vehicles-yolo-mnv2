from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf
import keras
from keras import layers
from keras.applications import MobileNetV2
from keras.callbacks import Callback

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import wandb
from wandb.integration.keras import WandbMetricsLogger



def build_datasets(dataset_path: Path,
                   img_size: int,
                   batch_size: int,
                   num_classes: int,
                   shuffle: bool = True,
                   seed: int = 777) -> tuple[tf.data.Dataset, tf.data.Dataset, 
                                             tf.data.Dataset, list[str]]:
    """Load train / val splits and apply preprocessing."""
    load_kwargs = dict(
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
 
    train_ds = keras.utils.image_dataset_from_directory(
        dataset_path / "train", **load_kwargs
    )
    val_ds = keras.utils.image_dataset_from_directory(
        dataset_path / "valid", shuffle=False,
        image_size=(img_size, img_size), batch_size=batch_size,
    )
 
    test_path = dataset_path / "test"
    test_ds = (
        keras.utils.image_dataset_from_directory(
            test_path, shuffle=False,
            image_size=(img_size, img_size), batch_size=batch_size,
        )
        if test_path.exists()
        else None
    )
 
    class_names = train_ds.class_names
 
    resize = (img_size, img_size)
 
    def preprocess(image, label):
        image = tf.image.resize(image, resize)
        label = tf.one_hot(label, num_classes)
        return image, label
 
    train_ds = train_ds.map(preprocess)
    val_ds   = val_ds.map(preprocess)
    if test_ds is not None:
        test_ds = test_ds.map(preprocess)
 
    return train_ds, val_ds, test_ds, class_names



# augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    # layers.RandomFlip("vertical"),
    # layers.RandomRotation(0.1),
    layers.RandomContrast(0.1)], 
    name="augmentation"
)

def build_model(num_classes: int, img_size: int, lr: float) -> keras.Model:
    inputs = layers.Input(shape=(img_size, img_size, 3))

    x = data_augmentation(inputs)
    x = layers.Rescaling(1. / 255)(x)

    backbone = MobileNetV2(include_top=False, input_tensor=x, weights="imagenet")
    backbone.trainable = False

    x = layers.GlobalAveragePooling2D(name="avg_pool")(backbone.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = keras.Model(inputs, outputs, name="MobileNetV2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=[
            keras.metrics.F1Score(average="macro", name="f1"),
            "accuracy",
        ]
    )
    return model



def unfreeze_model(model: keras.Model, lr: float) -> keras.Model:
    for layer in model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=[
            keras.metrics.F1Score(average="macro", name="f1"),
            "accuracy",
        ]
    )
    return model



def evaluate_on_test(model: keras.Model, 
                     test_ds: tf.data.Dataset, 
                     class_names: list[str], 
                     num_classes: int) -> None:
    """Evaluate model on test set and log results to wandb."""
    preds, labels = [], []
    for images, lbls in test_ds:
        preds.extend(np.argmax(model.predict(images, verbose=0), axis=1))
        labels.extend(np.argmax(lbls.numpy(), axis=1))

    # confusion matrix
    cm = confusion_matrix(labels, preds, labels=range(num_classes))
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
    cm_df.insert(0, "Class", class_names)
    cm_df = cm_df.reset_index(drop=True)
    
    # classification report
    report = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "Class"})

    wandb.log({
        "test_confusion_matrix": wandb.Table(dataframe=cm_df),
        "test_classification_report": wandb.Table(dataframe=report_df),
    })
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))



class ConfusionMatrixCallback(Callback):
    def __init__(self, val_data: tf.data.Dataset, class_names: list[str], num_classes: int):
        super().__init__()
        self.val_data = val_data
        self.class_names = class_names
        self.num_classes = num_classes

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        preds, labels = [], []
        for images, lbls in self.val_data:
            preds.extend(np.argmax(self.model.predict(images, verbose=0), axis=1))
            labels.extend(np.argmax(lbls.numpy(), axis=1))

        cm = confusion_matrix(labels, preds, labels=range(self.num_classes))
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_df = pd.DataFrame(cm_norm, index=self.class_names, columns=self.class_names)
        cm_df.insert(0, "Class", self.class_names)
        cm_df = cm_df.reset_index(drop=True)

        wandb.log({f"epoch_{epoch}_confusion_matrix": wandb.Table(dataframe=cm_df)})



def main(dataset_path: Path = Path("./dataset"),
         epochs: int = 100,
         img_size: int = 64,
         batch_size: int = 64,
         lr: float = 1e-5,
         num_classes: int = 42,
         patience: int = 10,
         result_dir: Path = Path("runs/classification")) -> keras.callbacks.History | None:
    """MobileNetV2 fine-tuning pipeline."""
    train_ds, val_ds, test_ds, class_names = build_datasets(
        dataset_path, img_size, batch_size, num_classes
    )

    model = build_model(num_classes=num_classes, img_size=img_size, lr=lr)
    model = unfreeze_model(model, lr=lr)
    model.summary()

    callbacks = [
        WandbMetricsLogger(log_freq="epoch"),
        ConfusionMatrixCallback(val_ds, class_names, num_classes),
        keras.callbacks.EarlyStopping(
            monitor="val_f1",
            patience=patience,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        )
    ]

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
    )

    evaluate_on_test(model, test_ds, class_names, num_classes)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"MobileNetV2_e{epochs}_{timestamp}.keras"
    save_path = result_dir / filename
    result_dir.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"Model saved -> {save_path}")

    return history


# main section
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MobileNetV2 training pipeline.")
    parser.add_argument("--dataset", type=Path, default="./dataset", help="Path to image dataset directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--img-size", type=int, default=224, dest="img_size", help="Input image size (square)")
    parser.add_argument("--batch-size", type=int, default=64, dest="batch_size", help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num-classes", type=int, default=6, dest="num_classes", help="Number of output classes")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--result-dir", type=Path, default="runs/classification", dest="result_dir", help="Directory to save trained model")
    parser.add_argument("--wandb-project", type=str, default="mobilenetv2-finetune", dest="wandb_project", help="W&B project name")
    parser.add_argument("--tag", type=str, default="mobilenetv2_ft", help="Run tag / experiment name")
    args = parser.parse_args()

    wandb.login()
    wandb.init(
        project=args.wandb_project,
        name=args.tag,
        config={
            "learning_rate": args.lr,
            "architecture": "MobileNetV2",
            "epochs": args.epochs,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
        },
    )

    main(dataset_path=args.dataset,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        lr=args.lr,
        num_classes=args.num_classes,
        patience=args.patience,
        result_dir=args.result_dir,
    )

    wandb.finish()