import torch
import pytorch_lightning as pl
from model_variants import calculate_class_weights, CNNFetalCLIPWithTextEmbeddings
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import GroupKFold
import argparse
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from dataset import VideoDataset, custom_collate

torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="./data/tensors",
        help="Directory containing processed tensor files",
    )
    parser.add_argument(
        "--annotations_path",
        type=str,
        default="./data/annotations.csv",
        help="Path to the annotations CSV file",
    )
    parser.add_argument(
        "--classification_type",
        type=str,
        default="binary",
        choices=["binary", "multiclass"],
        help="Type of classification task",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--pair_frames_with_text",
        action="store_true",
        help="Pair video frames with text prompts during training",
    )
    parser.add_argument(
        "--num_abnormalities",
        type=int,
        default=None,
        help="Number of abnormality classes to include (excluding 'Normal'). "
        "If not specified, all valid abnormalities will be used.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Initial learning rate"
    )
    parser.add_argument(
        "--fetalclip_checkpoint",
        type=str,
        default="./weights/fetalclip.pt",
        help="Path to FetalCLIP checkpoint (only for foundation model)",
    )
    return parser.parse_args()


def get_annotations(args):
    """Get annotations from CSV file and prepare labels based on classification type"""
    # Read annotations and filter out commented rows FIRST
    # as comments may reflect that label for abnormality is not reliable
    df = pd.read_csv(args.annotations_path)
    df = df[df["comments"].isna() | (df["comments"] == "")]
    df = df[df["view"] == "4CH"]  # Filter for 4CH view

    print(f"Total samples after initial filtering: {len(df)}")

    if args.classification_type == "binary":
        # Binary classification (normal vs abnormal)
        labels = (df["normal_abnormal"] == "Abnormal").astype(int)
        df["label"] = labels
        num_classes = 2
        class_names = ["Normal", "Abnormal"]

        print(f"Class distribution for binary classification:")
        print(f"Normal: {(labels == 0).sum()} samples")
        print(f"Abnormal: {(labels == 1).sum()} samples")

    elif args.classification_type == "multiclass":
        # Filter abnormalities with at least 10 examples
        # so that we have enough data for each class
        abnormality_counts = Counter(df[df["abnormality"].notna()]["abnormality"])
        sorted_abnormalities = sorted(
            abnormality_counts.items(), key=lambda x: x[1], reverse=True
        )

        if args.num_abnormalities:
            valid_abnormalities = [
                abn for abn, count in sorted_abnormalities[: args.num_abnormalities]
            ]
        else:
            valid_abnormalities = [k for k, v in abnormality_counts.items() if v >= 10]

        # Create mapping of abnormalities to integer labels
        abnormality_to_label = {
            abn: idx + 1 for idx, abn in enumerate(valid_abnormalities)
        }

        # Create labels and filter dataframe
        labels = []
        valid_indices = []

        for idx, row in df.iterrows():
            # abnormality class is empty and normal_abnormal column is normal
            if pd.isna(row["abnormality"]) and row["normal_abnormal"] == "Normal":
                labels.append(0)
                valid_indices.append(idx)
            elif (
                not pd.isna(row["abnormality"])
                and row["abnormality"] in valid_abnormalities
            ):
                labels.append(abnormality_to_label[row["abnormality"]])
                valid_indices.append(idx)

        # Filter dataframe to only include valid samples
        df = df.loc[valid_indices].copy()
        labels = np.array(labels)
        df["label"] = labels

        # valid abnormalities + normal
        num_classes = len(valid_abnormalities) + 1
        class_names = ["Normal"] + valid_abnormalities

        print(f"\nClass distribution after filtering:")
        for i, name in enumerate(class_names):
            count = (labels == i).sum()
            print(f"{name}: {count} samples")

    else:
        raise ValueError(f"Invalid classification_type: {args.classification_type}")

    # Group by patient ID for stratification
    df["pid"] = df["video"].apply(lambda x: os.path.basename(x).split("_")[0])

    # Initialize group k-fold
    group_kfold = GroupKFold(n_splits=5)

    # Store fold information
    df["fold"] = -1
    for fold_idx, (train_idx, val_idx) in enumerate(
        group_kfold.split(df, df["label"], groups=df["pid"])
    ):
        df.iloc[val_idx, df.columns.get_loc("fold")] = fold_idx

    fold_metrics = []

    # Create directories for saving results
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    for fold in range(5):
        print(f"\nPreparing Fold {fold+1}")

        # Get train/val splits
        train_df = df[df["fold"] != fold].copy()
        val_df = df[df["fold"] == fold].copy()

        if args.classification_type == "multiclass":
            # Oversample training data for multiclass
            ros = RandomOverSampler(random_state=args.seed)
            train_indices = np.arange(len(train_df))

            print(f"\nClass distribution BEFORE oversampling (Fold {fold+1}):")
            for i, name in enumerate(class_names):
                count = (train_df["label"] == i).sum()
                print(f"{name}: {count} samples")

            train_indices_resampled, _ = ros.fit_resample(
                train_indices.reshape(-1, 1), train_df["label"]
            )
            train_df = train_df.iloc[train_indices_resampled.ravel()].reset_index(
                drop=True
            )

            print(f"\nClass distribution AFTER oversampling (Fold {fold+1}):")
            for i, name in enumerate(class_names):
                count = (train_df["label"] == i).sum()
                print(f"{name}: {count} samples")
        fold_data = {
            "train": train_df,
            "val": val_df,
            "num_classes": num_classes,
            "class_names": class_names,
        }

        fold_metrics.append(fold_data)

    return {
        "folds": fold_metrics,
        "class_names": class_names,
        "num_classes": num_classes,
    }


def train_fold(fold_data, fold, args):
    """Train model for one fold"""
    train_df = fold_data["train"]
    val_df = fold_data["val"]
    num_classes = fold_data["num_classes"]
    class_names = fold_data["class_names"]
    # Calculate class weights using improved method
    labels = train_df["label"].values
    class_weights = calculate_class_weights(labels, num_classes)
    print(f"\nClass weights for fold {fold}:")
    for i in range(num_classes):
        print(f"Class {i}: {class_weights[i]:.4f}")
    # Create datasets
    train_dataset = VideoDataset(
        df=train_df,
        video_dir=args.processed_dir,
        split="train",
        num_classes=num_classes,
        model_type="foundation",
        pair_frames_with_text=args.pair_frames_with_text,
    )

    val_dataset = VideoDataset(
        df=val_df,
        video_dir=args.processed_dir,
        split="val",
        num_classes=num_classes,
        model_type="foundation",
        pair_frames_with_text=args.pair_frames_with_text,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate,
    )
    print("\n--- Train Loader Batch ---")
    for batch in train_loader:
        videos, true_labels, labels, names, text_ids, neg_idx = batch
        print(f"Videos shape: {videos.shape}")
        print(
            f"Real Labels shape:  {true_labels.shape}, Labels: {true_labels.tolist()}"
        )
        print(f"Binary Labels shape: {labels.shape}, Labels: {labels.tolist()}")
        print(f"Names: {names[:2]}...")
        print(f"Positive/text: {text_ids.tolist()}")
        print(f"Negative: {neg_idx.tolist()}")

        break

    print("\n--- Validation Loader Batch ---")
    for batch in val_loader:
        videos, true_labels, labels, names, text_ids, neg_idx = batch
        print(f"Videos shape: {videos.shape}")
        print(
            f"Real Labels shape:  {true_labels.shape}, Labels: {true_labels.tolist()}"
        )
        print(f"Vinary Labels shape: {labels.shape}, Labels: {labels.tolist()[:10]}")
        print(f"Names: {names[:2]}...")
        print(f"Positive/text: {text_ids.tolist()}")
        print(f"Negative: {neg_idx.tolist()}")
        break
    selected_prompt_indices = {
        "Normal": [0, 1],
        "Abnormal": [0, 1],
    }
    model_params = {
        "num_classes": num_classes,
        "learning_rate": args.learning_rate,
        "class_weights": class_weights,
        "encoder_checkpoint": args.fetalclip_checkpoint,
        "backmodel": "fetalclip",
        "selected_prompt_indices": selected_prompt_indices,
    }
    model = CNNFetalCLIPWithTextEmbeddings(class_names=class_names, **model_params)

    early_stopping = pl.callbacks.EarlyStopping(
        monitor=(
            "val_f1_macro" if num_classes > 2 else "val_f1"
        ),  # Use macro F1 for multiclass
        mode="max",
        patience=10,
        verbose=True,
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"checkpoints/fold_{fold}",
        filename=f"model_fold_{fold}",
        monitor="val_f1_macro" if num_classes > 2 else "val_f1",
        mode="max",
        save_top_k=1,
    )

    csv_logger = pl.loggers.CSVLogger(save_dir="logs", name=f"fold_{fold}")

    trainer = pl.Trainer(
        max_epochs=100,
        # accumulate_grad_batches=8,
        accelerator="cuda",
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=csv_logger,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, val_loader)
    return {
        "val_f1": checkpoint_callback.best_model_score.item(),
        "best_model_path": checkpoint_callback.best_model_path,
        "status": "success",
    }


def main():
    args = parse_args()

    # Validate input paths
    if not os.path.exists(args.processed_dir):
        raise ValueError(f"Processed directory {args.processed_dir} does not exist")
    if not os.path.exists(args.annotations_path):
        raise ValueError(f"Annotations file {args.annotations_path} does not exist")

    pl.seed_everything(args.seed)

    data = get_annotations(args)

    for fold, fold_split in enumerate(data["folds"]):
        print(f"\nTraining Fold {fold+1}")
        print(f"Class distribution in training set:")
        for i, name in enumerate(data["class_names"]):
            count = (fold_split["train"]["label"] == i).sum()
            print(f"{name}: {count} samples")

        train_fold(fold_split, fold, args)

if __name__ == "__main__":
    main()
