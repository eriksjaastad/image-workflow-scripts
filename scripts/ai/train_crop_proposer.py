#!/usr/bin/env python3
"""
Phase 2.4: Train Crop Proposer Model

This model predicts optimal crop coordinates (x1, y1, x2, y2) for a selected image.

Training data: select_crop_log.csv (7,194 examples from Mojo 1 & 2)
Input: CLIP embedding (512-dim) + image dimensions (2-dim) = 514-dim
Output: Normalized crop coordinates [x1, y1, x2, y2] in range [0, 1]

Architecture: MLP regression model
Loss: MSE (Mean Squared Error) on normalized coordinates
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class CropExample:
    """Single crop training example."""

    image_path: str
    embedding: np.ndarray
    width: int
    height: int
    crop_x1: float
    crop_y1: float
    crop_x2: float
    crop_y2: float


class CropDataset(Dataset):
    """Dataset for crop coordinate prediction."""

    def __init__(self, examples: list[CropExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Input: embedding + normalized dimensions
        embedding = torch.from_numpy(ex.embedding).float()
        dims = torch.tensor(
            [ex.width / 2048.0, ex.height / 2048.0], dtype=torch.float32
        )
        x = torch.cat([embedding, dims])

        # Output: normalized crop coordinates [0, 1]
        y = torch.tensor(
            [
                ex.crop_x1 / ex.width,
                ex.crop_y1 / ex.height,
                ex.crop_x2 / ex.width,
                ex.crop_y2 / ex.height,
            ],
            dtype=torch.float32,
        )

        return x, y


class CropProposerModel(nn.Module):
    """MLP model to predict crop coordinates."""

    def __init__(self, input_dim=514, hidden_dims=None):
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3)])
            prev_dim = hidden_dim

        # Output layer: 4 coordinates (x1, y1, x2, y2)
        layers.append(nn.Linear(prev_dim, 4))
        layers.append(nn.Sigmoid())  # Ensure [0, 1] range

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def load_embeddings(cache_file: Path) -> dict[str, np.ndarray]:
    """Load all embeddings from cache."""
    embeddings = {}

    with open(cache_file) as f:
        for line in f:
            data = json.loads(line)
            img_path = data["image_path"]
            emb_file = Path(data["embedding_file"])

            if emb_file.exists():
                embeddings[img_path] = np.load(emb_file)

    return embeddings


def load_crop_training_data(
    log_file: Path, embeddings: dict[str, np.ndarray]
) -> list[CropExample]:
    """Load crop training data from CSV log."""
    examples = []
    skipped = 0

    with open(log_file) as f:
        reader = csv.DictReader(f)

        for row in reader:
            chosen_path = row["chosen_path"].strip()

            # Try to find embedding by filename
            filename = Path(chosen_path).name
            embedding = None

            for emb_path, emb in embeddings.items():
                if Path(emb_path).name == filename:
                    embedding = emb
                    break

            if embedding is None:
                skipped += 1
                continue

            # Parse crop coordinates
            try:
                crop_x1 = float(row["crop_x1"])
                crop_y1 = float(row["crop_y1"])
                crop_x2 = float(row["crop_x2"])
                crop_y2 = float(row["crop_y2"])

                # Get image dimensions (use chosen image dims)
                # Try width_1/height_1 first (chosen image), fallback to width_0/height_0
                if row.get("width_1") and int(row["width_1"]) > 0:
                    width = int(row["width_1"])
                    height = int(row["height_1"])
                elif row.get("width_0") and int(row["width_0"]) > 0:
                    width = int(row["width_0"])
                    height = int(row["height_0"])
                else:
                    # Default dimensions if not available
                    width = 2048
                    height = 2048

                # Skip invalid crops
                if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                    skipped += 1
                    continue

                examples.append(
                    CropExample(
                        image_path=chosen_path,
                        embedding=embedding,
                        width=width,
                        height=height,
                        crop_x1=crop_x1,
                        crop_y1=crop_y1,
                        crop_x2=crop_x2,
                        crop_y2=crop_y2,
                    )
                )

            except (ValueError, KeyError):
                skipped += 1
                continue

    return examples


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Validating", leave=False):
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    # Paths
    data_dir = Path("data")
    cache_file = data_dir / "ai_data" / "cache" / "processed_images.jsonl"
    log_file = data_dir / "training" / "select_crop_log.csv"
    model_dir = data_dir / "ai_data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    batch_size = 32
    epochs = 50
    lr = 0.001
    val_split = 0.15

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load data
    embeddings = load_embeddings(cache_file)
    examples = load_crop_training_data(log_file, embeddings)

    if len(examples) == 0:
        return

    # Split train/val
    np.random.seed(42)
    np.random.shuffle(examples)

    val_size = int(len(examples) * val_split)
    train_examples = examples[val_size:]
    val_examples = examples[:val_size]

    # Create datasets
    train_dataset = CropDataset(train_examples)
    val_dataset = CropDataset(val_examples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    model = CropProposerModel(input_dim=514).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                model_dir / "crop_proposer_v1.pt",
            )

    # Save metadata
    metadata = {
        "version": "v1",
        "model": "CropProposerModel",
        "input_dim": 514,
        "output_dim": 4,
        "hidden_dims": [512, 256, 128],
        "training_examples": len(train_examples),
        "validation_examples": len(val_examples),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs_trained": epochs,
        "device": str(device),
    }

    with open(model_dir / "crop_proposer_v1_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
