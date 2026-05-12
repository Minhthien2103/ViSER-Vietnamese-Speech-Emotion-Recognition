import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.Bi_LSTM import BiLSTMModel
from utils.dl_data_loader import get_dl_dataloaders
from evaluation.evaluate import evaluate
from evaluation.plotter import plot_training_history, plot_class_distribution


FEATURE_SOURCE = "librosa"  # Thay bằng "wav2vec" nếu muốn dùng Wav2Vec2
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
EARLY_STOP_PATIENCE = 10

MODEL_DIR = config.BASE_DIR / "models" / "saved"
MODEL_DIR.mkdir(parents = True, exist_ok = True)


def train_bilstm(source="librosa"):
    train_loader, val_loader, test_loader, num_features, num_meta, num_classes = get_dl_dataloaders(
        source=source, batch_size=BATCH_SIZE
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining Bi-LSTM on {device}...")

    # Khởi tạo model
    model = BiLSTMModel(
        input_size = num_features, 
        hidden_size = 128, 
        num_layers = 2, 
        num_meta = num_meta, 
        num_classes = num_classes
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr = LR, weight_decay = 1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.5, patience = 5)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = MODEL_DIR / f"bilstm_{source}.pth"

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
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss_avg = train_loss / train_total
        train_acc_avg = train_correct / train_total

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

        print(f"Epoch {epoch:02d}: Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc_avg:.4f} | Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc_avg:.4f}")

        scheduler.step(val_acc_avg)

        # Lưu best model & Early Stopping
        if val_acc_avg > best_val_acc:
            best_val_acc = val_acc_avg
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f" Model improved! Saved to {best_model_path.name}")
        else:
            patience_counter += 1
            print(f" Early stopping counter: {patience_counter}/{EARLY_STOP_PATIENCE}")
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping triggered! No improvement for {EARLY_STOP_PATIENCE} epochs.")
                break

    print(f"\nTraining complete! Best Val Acc: {best_val_acc:.4f}")
    
    # Plot curve
    plot_training_history(history, "BiLSTM", source)

    # EVALUATION TRÊN TẬP TEST TÁCH BIỆT
    print("\nĐánh giá trên tập TEST...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
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

    # Đánh giá và sinh biểu đồ Confusion Matrix
    evaluate(np.array(all_labels), np.array(all_preds), "BiLSTM", source)


if __name__ == "__main__":
    train_bilstm(source=FEATURE_SOURCE)
