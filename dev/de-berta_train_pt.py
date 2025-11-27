import copy
import time
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification
import pickle

# --- Configuration & Paths ---
OUTPUT_DIR = "deberta_trained"
DATA_PATH = "../jigsaw/dataset_text_target.csv"  # Ensure this path exists relative to where you run the script
PICKLE_LOCATION = "../pickles"

MODEL_NAME = "microsoft/deberta-v3-base"

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)




# --- Parameters ---
MAXIMUM_LENGTH = 256
BATCH_SIZE = 32
EPOCHS = 50
THRESHOLD = 0.5
# DATASET_SIZE = 1_000

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if (torch.backends.mps.is_available()) else "cpu")
print(f"Using device: {DEVICE}")

def weighted_average(nums, weights):
    if sum(weights) == 0: return 0
    return sum(x[0] * x[1] for x in zip(nums, weights)) / sum(weights)

def format_time(seconds):
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    res = ""
    if h != 0: res += f"{h}h"
    if m != 0: res += f", {m}m" if h != 0 else f"{m}m"
    if s != 0: res += f", {s}s" if h != 0 or m != 0 else f"{s}s"
    return res

def load_and_process_data():
    print("Loading and processing data...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH).dropna()
    # df_true = df[df.target > 0.5]
    # df_false = df[df.target <= 0.5]

    # Balance the dataset
    # df = pd.concat([df_true[:DATASET_SIZE // 2], df_false[:DATASET_SIZE // 2]], axis=0)

    mapper = lambda x: 1 if x > THRESHOLD else 0
    df['target'] = df['target'].apply(mapper)

    return train_test_split(
        df.comment_text, df.target, test_size=0.2, random_state=42, shuffle=True
    )

def main():
    # 1. Prepare Data
    x_train, x_test, y_train, y_test = load_and_process_data()


    print("Loading training encodings")
    with open(os.path.join(PICKLE_LOCATION, f"X_train_encoded.pkl"), "rb") as f:
        X_train_encoded = pickle.load(f)
    print("Loading testing encodings")
    with open(os.path.join(PICKLE_LOCATION, f"X_test_encoded.pkl"), "rb") as f:
        X_test_encoded = pickle.load(f)
    print("Loaded encodings!")

    # 3. Create Datasets & Loaders
    train_dataset = TensorDataset(
        X_train_encoded['input_ids'],
        X_train_encoded['attention_mask'],
        torch.tensor(y_train.values, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        X_test_encoded['input_ids'],
        X_test_encoded['attention_mask'],
        torch.tensor(y_test.values, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Initialize Model
    print("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    model.to(DEVICE)

    # Note: The notebook had freezing code commented out.
    # If you wish to freeze layers, uncomment the following logic:
    # for param in model.deberta.parameters():
    #     param.requires_grad = False
    # for param in model.classifier.parameters():
    #     param.requires_grad = True

    # 5. Optimizer & Loss
    criterion = nn.BCEWithLogitsLoss()
    # Filter only parameters that require gradients (in case freezing is enabled)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # History tracking
    history = {
        'loss': [],
        'val_loss': [],
        'accuracy': [],
        'val_accuracy': [],
    }

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    print("Starting training...")
    for epoch in range(EPOCHS):
        start_time = time.time()

        # --- Training Phase ---
        model.train()
        total_loss = 0
        accuracies = []
        sizes = []

        for i, (input_ids, attention_mask, labels) in enumerate(train_loader):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            print(f"\rBatch {i + 1} of {len(train_loader)}", end="")

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)[0]

            # Calculate metrics
            preds_prob = torch.sigmoid(outputs).cpu()
            batch_acc = accuracy_score(preds_prob > THRESHOLD, labels.cpu())
            accuracies.append(batch_acc)
            sizes.append(len(labels))

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("") # Newline after batch printing
        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = weighted_average(accuracies, sizes)

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        val_accuracies = []
        val_sizes = []

        with torch.no_grad():
            for input_ids, attention_mask, labels in test_loader:
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)

                outputs = model(input_ids, attention_mask=attention_mask)[0]

                preds_prob = torch.sigmoid(outputs).cpu()
                batch_acc = accuracy_score(preds_prob > THRESHOLD, labels.cpu())
                val_accuracies.append(batch_acc)
                val_sizes.append(len(labels))

                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_val_acc = weighted_average(val_accuracies, val_sizes)
        avg_val_loss = total_val_loss / len(test_loader)

        # Update History
        history['loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['accuracy'].append(avg_train_acc)
        history['val_accuracy'].append(avg_val_acc)

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        end_time = time.time()
        print(f"Epoch {epoch + 1}/{EPOCHS} | "
              f"Train Acc: {avg_train_acc:.4f} | "
              f"Val Acc: {avg_val_acc:.4f} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Time Taken: {format_time(end_time - start_time)}\n")

    # Load best weights
    model.load_state_dict(best_model_wts)

    # 6. Final Evaluation
    print("Running final evaluation on test set...")
    model.eval()
    y_pred_prob = []
    y_true_all = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            outputs = torch.sigmoid(model(input_ids, attention_mask=attention_mask)[0]).squeeze()

            # Handle edge case where batch size is 1 (squeeze removes too many dims)
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)

            y_pred_prob.extend(outputs.cpu().numpy())
            y_true_all.extend(labels.cpu().numpy())

    y_pred = (np.array(y_pred_prob) > THRESHOLD).astype(int)

    print('\nEvaluation Metrics:')
    print('Accuracy:', accuracy_score(y_true_all, y_pred))
    print('Precision:', precision_score(y_true_all, y_pred))
    print('Recall:', recall_score(y_true_all, y_pred))
    print('F1-score:', f1_score(y_true_all, y_pred))
    print('\nClassification Report:')
    print(classification_report(y_true_all, y_pred))

    # --- Saving Outputs ---

    # 1. Save History JSON
    history_path = os.path.join(OUTPUT_DIR, "history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")

    # 2. Save Plots
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Train")
    plt.plot(history["val_accuracy"], label="Validation")
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "training_plots.png")
    plt.savefig(plot_path)
    print(f"Plots saved to {plot_path}")
    plt.close() # Close to free memory

    # 3. Save Model
    model_save_path = os.path.join(OUTPUT_DIR, "deberta_trained_pytorch.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()