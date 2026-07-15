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

PICKLE_LOCATION = "../pickles"

# Load and preprocess the dataset
# Make sure the path to your CSV is correct
print("Reading dataset...")
df = pd.read_csv("../jigsaw/dataset_text_target.csv").dropna()
print("Dataset size:", len(df))
# df = df[:100]

os.makedirs(PICKLE_LOCATION, exist_ok=True)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if (torch.mps and torch.mps.is_available()) else "cpu")
print("Using device:", DEVICE)


model_name = "microsoft/deberta-v3-base"

# Tokenize and encode the data using the BERT tokenizer
print("Loading tokenizer:", model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True, use_fast=False)
print("Loaded tokenizer!")

pickle_file = os.path.abspath(os.path.join(PICKLE_LOCATION, "comment_text_encoded.pt"))

print("Pickle file:", pickle_file)

start = time.time()
print("Encoding dataset...")
X_train_encoded = tokenizer.batch_encode_plus(
    df["comment_text"].tolist(),
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

torch.save(X_train_encoded, pickle_file)

with open(pickle_file, "wb") as f:
    pickle.dump(X_train_encoded, f)
print("Written to disk!")
print(f"Load using `torch.load({pickle_file}, weights_only=False)`")
