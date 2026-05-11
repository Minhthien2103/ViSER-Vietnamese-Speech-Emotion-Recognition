import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.CNN import CNN1DModel
from utils.dl_data_loader import get_dl_dataloaders
from evaluation.evaluate import evaluate
from evaluation.plotter import plot_training_history, plot_class_distribution


FEATURE_SOURCE = "librosa"  # Thay bằng "wav2vec" nếu muốn dùng Wav2Vec2
EPOCHS = 80
BATCH_SIZE = 64
LR = 3e-4
EARLY_STOP_PATIENCE = 15
LABEL_SMOOTHING = 0.1

MODEL_DIR = config.BASE_DIR / "models" / "saved"
MODEL_DIR.mkdir(parents = True, exist_ok = True)


def train_1dcnn(source = "librosa"):
    train_loader, val_loader, test_loader, num_features, num_meta, num_classes = get_dl_dataloaders(
        source = source, batch_size = BATCH_SIZE
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining CNN on {device}...")

    model = CNN1DModel(
        input_channels = num_features,
        num_meta = num_meta,
        num_classes = num_classes,
        dropout = 0.3,
    ).to(device)

    # Tổng số tham số
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # Label smoothing giúp model không quá tự tin, giảm overfit
    criterion = nn.CrossEntropyLoss(label_smoothing = LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr = LR, weight_decay = 1e-3)

    # CosineAnnealing — giảm LR mượt hơn ReduceLROnPlateau
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0 = 10, T_mult = 2, eta_min = 1e-6
    )

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = MODEL_DIR / f"cnn1d_{source}.pth"

    for epoch in range(1, EPOCHS + 1):
        # TRAIN
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch in tqdm(train_loader, desc = f"Epoch {epoch:02d}/{EPOCHS} [Train]", leave = False):
            features = batch[0].to(device)
            labels = batch[-1].to(device)
            meta = batch[1].to(device) if len(batch) == 3 else None

            optimizer.zero_grad()
            outputs = model(features, meta)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping chống gradient exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

            optimizer.step()

            train_loss += loss.item() * features.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss_avg = train_loss / train_total
        train_acc_avg = train_correct / train_total

        scheduler.step()

        # VALIDATION
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc = f"Epoch {epoch:02d}/{EPOCHS} [Val  ]", leave = False):
                features = batch[0].to(device)
                labels = batch[-1].to(device)
                meta = batch[1].to(device) if len(batch) == 3 else None

                outputs = model(features, meta)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * features.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss_avg = val_loss / val_total
        val_acc_avg = val_correct / val_total

        history['train_loss'].append(train_loss_avg)
        history['val_loss'].append(val_loss_avg)
        history['train_acc'].append(train_acc_avg)
        history['val_acc'].append(val_acc_avg)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}: Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc_avg:.4f} | Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc_avg:.4f} | LR: {current_lr:.6f}")

        # Lưu best model & Early Stopping
        if val_acc_avg > best_val_acc:
            best_val_acc = val_acc_avg
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  Model improved! Saved to {best_model_path.name}")
        else:
            patience_counter += 1
            print(f"  Early stopping counter: {patience_counter}/{EARLY_STOP_PATIENCE}")
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping triggered! No improvement for {EARLY_STOP_PATIENCE} epochs.")
                break

    print(f"\nTraining complete! Best Val Acc: {best_val_acc:.4f}")

    # Plot curve
    plot_training_history(history, "CNN1D", source)

    # EVALUATION TRÊN TẬP TEST TÁCH BIỆT
    print("\nĐánh giá trên tập TEST...")
    model.load_state_dict(torch.load(best_model_path, map_location = device))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            features = batch[0].to(device)
            labels = batch[-1].numpy()
            meta = batch[1].to(device) if len(batch) == 3 else None

            outputs = model(features, meta)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels)

    # Dùng hàm evaluate chung (sẽ in report + plot matrix)
    evaluate(np.array(all_labels), np.array(all_preds), "CNN1D", source)


if __name__ == "__main__":
    train_1dcnn(source = FEATURE_SOURCE)
