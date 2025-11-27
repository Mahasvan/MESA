import time

print("Importing libraries...")
import os
import pickle

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

print("Imported libraries!")

maximum_length = 256  # Max length of input sequences

threshold = 0.5


PICKLE_LOCATION = "../pickles"

# Load and preprocess the dataset
# Make sure the path to your CSV is correct
print("Reading dataset...")
df = pd.read_csv("../jigsaw/dataset_text_target.csv").dropna()
print("Dataset size:", len(df))
# df_true = df[df.target > 0.5]
# df_false = df[df.target <= 0.5]
# df = pd.concat([df_true[:DATASET_SIZE // 2], df_false[:DATASET_SIZE // 2]], axis=0)
mapper = lambda x: 1 if x > 0.5 else 0
df['target'] = df['target'].apply(mapper)
print("Read dataset!")

os.makedirs(PICKLE_LOCATION, exist_ok=True)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    df.comment_text, df.target, test_size=0.2, random_state=42, shuffle=True
)
print("Split dataset into train and test!")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if (torch.mps and torch.mps.is_available()) else "cpu")
print("Using device:", DEVICE)


model_name = "microsoft/deberta-v3-base"


# Tokenize and encode the data using the BERT tokenizer
print("Loading tokenizer:", model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True, use_fast=False)
print("Loaded tokenizer!")

train_file = os.path.join(PICKLE_LOCATION, "X_train_encoded.pkl")
test_file = os.path.join(PICKLE_LOCATION, "X_test_encoded.pkl")

print("Train file:", train_file)
print("Test file:", test_file)

start = time.time()
print("Encoding train...")
X_train_encoded = tokenizer.batch_encode_plus(
    x_train.tolist(),
    padding='max_length',
    truncation=True,
    max_length=maximum_length,
    add_special_tokens=True,
    return_tensors='pt',  # Return PyTorch tensors
    verbose=True
)
end = time.time()
print(f"Encoded train file! Time Taken: {int(end - start)} seconds")

print("Writing to disk...")
with open(train_file, "wb") as f:
    pickle.dump(X_train_encoded, f)
print("Written to disk!")

print("Encoding test...")
start = time.time()
X_test_encoded = tokenizer.batch_encode_plus(
    x_test.tolist(),
    padding='max_length',
    truncation=True,
    max_length=maximum_length,
    add_special_tokens=True,
    return_tensors='pt',  # Return PyTorch tensors
    verbose=True
)
end = time.time()
print(f"Encoded test file! Time Taken: {int(end - start)} seconds")

print("Writing to disk...")
with open(test_file, "wb") as f:
    pickle.dump(X_test_encoded, f)
print("Written to disk!")