import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from transformers import BertTokenizer
from dynamic_dataloader import CustomDataLoader
import math
import os
from datetime import datetime

# Setting up the parameters
maximum_features = 30522  # Vocabulary size for BERT
maximum_length = 128  # Max length of input sequences
word_embedding_dims = 50  # Dimension of word embeddings (Note: In PyTorch this is embedding_dim)
no_of_filters = 128  # Number of filters for Conv1D
kernel_size = 3  # Size of the convolutional kernel
hidden_dim_1 = 128  # Neurons in the dense hidden layer

gamma = 0.2

batch_size = 64
epochs = 25
threshold = 0.7
# DATASET_SIZE = 100_000

# Load and preprocess the dataset
# Make sure the path to your CSV is correct
df = pd.read_csv("../jigsaw/dataset_text_target.csv").dropna()
# df_true = df[df.target > 0.5]
# df_false = df[df.target <= 0.5]
# df = pd.concat([df_true[:DATASET_SIZE // 2], df_false[:DATASET_SIZE // 2]], axis=0)
mapper = lambda x: 1 if x > 0.5 else 0
df['target'] = df['target'].apply(mapper)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    df.comment_text, df.target, test_size=0.2, random_state=42, shuffle=True
)
# Tokenize and encode the data using the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Encode the training and test data
X_train_encoded = tokenizer.batch_encode_plus(
    x_train.tolist(),
    padding='max_length',
    truncation=True,
    max_length=maximum_length,
    add_special_tokens=True,
    return_tensors='pt',  # Return PyTorch tensors
)

X_test_encoded = tokenizer.batch_encode_plus(
    x_test.tolist(),
    padding='max_length',
    truncation=True,
    max_length=maximum_length,
    add_special_tokens=True,
    return_tensors='pt',  # Return PyTorch tensors
)

# Create PyTorch Datasets
train_dataset = TensorDataset(X_train_encoded['input_ids'], torch.tensor(y_train.values, dtype=torch.float32))
test_dataset = TensorDataset(X_test_encoded['input_ids'], torch.tensor(y_test.values, dtype=torch.float32))

# Create DataLoaders
train_loader = CustomDataLoader(train_dataset, batch_size=batch_size, gamma=gamma)
test_loader = CustomDataLoader(test_dataset, batch_size=batch_size, gamma=gamma)

class CNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_size, hidden_dim):
        super(CNNTextClassifier, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Convolutional layers
        self.conv1 = nn.Conv1d(embedding_dim, n_filters, kernel_size=filter_size, padding='valid')
        self.pool1 = nn.MaxPool1d(kernel_size=3)

        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=filter_size, padding='valid')
        self.pool2 = nn.MaxPool1d(kernel_size=3)

        self.conv3 = nn.Conv1d(n_filters, n_filters, kernel_size=filter_size, padding='valid')
        # Global Max Pooling is achieved with AdaptiveMaxPool1d
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # Dense layers
        self.fc1 = nn.Linear(n_filters, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids):
        # input_ids shape: (batch_size, seq_len)
        embedded = self.embedding(input_ids)
        # embedded shape: (batch_size, seq_len, embedding_dim)

        # PyTorch Conv1d expects (batch_size, channels, seq_len)
        # So we permute the dimensions
        embedded = embedded.permute(0, 2, 1)

        x = self.pool1(F.relu(self.conv1(embedded)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))

        x = self.global_pool(x).squeeze(2)  # Squeeze to remove the last dimension

        x = F.relu(self.fc1(x))
        output = torch.sigmoid(self.fc2(x))

        return output


# Instantiate the model
model = CNNTextClassifier(
    vocab_size=maximum_features,
    embedding_dim=word_embedding_dims,
    n_filters=no_of_filters,
    filter_size=kernel_size,
    hidden_dim=hidden_dim_1
)

# Set device (use GPU if available)
device = torch.device("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters())

# To store history
history = {
    'loss': [],
    'val_loss': [],
    'accuracy': [],
    'val_accuracy': []
}

best_val_loss = float('inf')
best_val_accuracy = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(epochs):
    # --- Training Phase ---
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0



    dynamic = True if epoch % 2 != 0 else False
    # print("Dynamic:", dynamic)
    times = 1
    if dynamic:
        times = math.ceil(1 / gamma)
    # print("Running", times, "times in this epoch")
    for _ in range(times):
        # do this batch loop how many ever times it needs
        # this will run 5 times, if its an even epoch, and gamma = 0.2
        batch_losses = {}
        for batch_id, (input_ids, labels) in train_loader.generate(dynamic=dynamic):
            # print(batch_id)
            input_ids, labels = input_ids.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)

            batch_losses[batch_id] = loss.item()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = (outputs > threshold).int()
            total_train += labels.size(0)
            correct_train += (predicted == labels.int()).sum().item()
        train_loader.update_losses(batch_losses)

    avg_train_loss = total_loss / len(train_loader)
    train_accuracy = correct_train / total_train

    # --- Validation Phase ---
    model.eval()
    total_val_loss = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for batch_id, (input_ids, labels) in test_loader.generate():
            input_ids, labels = input_ids.to(device), labels.to(device).unsqueeze(1)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

            predicted = (outputs > threshold).int()
            total_val += labels.size(0)
            correct_val += (predicted == labels.int()).sum().item()

    avg_val_loss = total_val_loss / len(test_loader)
    val_accuracy = correct_val / total_val

    # Save history
    history['loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['accuracy'].append(train_accuracy)
    history['val_accuracy'].append(val_accuracy)

    # Save the best model
    if val_accuracy < best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_wts = copy.deepcopy(model.state_dict())

    print(f"{'Dyn' if dynamic else 'Nor'} Epoch {epoch + 1}/{epochs} | "
          f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

# Load best model weights
model.load_state_dict(best_model_wts)

model.eval()
y_pred_prob = []
y_true = []

with torch.no_grad():
    for batch_id, (input_ids, labels) in test_loader.generate():
        input_ids = input_ids.to(device)
        outputs = model(input_ids).squeeze()
        y_pred_prob.extend(outputs.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

y_pred = (np.array(y_pred_prob) > threshold).astype(int)

# Calculating and printing evaluation metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
report_text = classification_report(y_true, y_pred)
report_dict = classification_report(y_true, y_pred, output_dict=True)

print('\nEvaluation Metrics:')
print('Accuracy:', acc)
print('Precision:', prec)
print('Recall:', rec)
print('F1-score:', f1)
print('\nClassification Report:')
print(report_text)

# Prepare results directory and filenames
results_dir = os.path.join('..', 'results')
os.makedirs(results_dir, exist_ok=True)
_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

# Save metrics and reports
import json
metrics_out = {
    'accuracy': float(acc),
    'precision': float(prec),
    'recall': float(rec),
    'f1_score': float(f1),
    'threshold': float(threshold),
    'epochs': int(epochs),
    'batch_size': int(batch_size)
}
with open(os.path.join(results_dir, f'cnn_dynamic_metrics_{_timestamp}.json'), 'w') as f:
    json.dump({'metrics': metrics_out, 'classification_report': report_dict}, f, indent=2)
with open(os.path.join(results_dir, f'cnn_dynamic_classification_report_{_timestamp}.txt'), 'w') as f:
    f.write(report_text)
# Also save raw predictions and labels for further analysis
np.save(os.path.join(results_dir, f'cnn_dynamic_y_true_{_timestamp}.npy'), np.array(y_true))
np.save(os.path.join(results_dir, f'cnn_dynamic_y_pred_{_timestamp}.npy'), np.array(y_pred))

# Save training history
with open(os.path.join(results_dir, f'cnn_dynamic_history_{_timestamp}.json'), 'w') as f:
    json.dump(history, f, indent=2)

# Plotting Model Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history["accuracy"])
plt.plot(history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")

# Plotting Model Loss
plt.subplot(1, 2, 2)
plt.plot(history["loss"])
plt.plot(history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")

plt.tight_layout()
plot_path = os.path.join(results_dir, f'cnn_dynamic_training_curves_{_timestamp}.png')
plt.savefig(plot_path, dpi=200)
plt.close()

torch.save(model.state_dict(), "../cnn_model_trained_torch/cnn_pytorch_dynamic.pth")
print("Model saved to cnn_pytorch_dynamic.pth")
print(f"Saved metrics and plots to: {results_dir}")