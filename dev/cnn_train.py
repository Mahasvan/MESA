# importing the necessary libraries
import pandas as pd
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, MaxPooling1D
from keras.models import Sequential
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Setting up the parameters
maximum_features = 30522  # Maximum number of words to consider as features
maximum_length = 128  # Maximum length of input sequences
word_embedding_dims = 50  # Dimension of word embeddings
no_of_filters = 128  # Number of filters in the convolutional layer
kernel_size = 3  # Size of the convolutional filters
hidden_dim_1 = 128  # Number of neurons in the hidden layer

batch_size = 64  # Batch size for training
epochs = 10  # Number of training epochs
threshold = 0.5  # Threshold for binary classification

DATASET_SIZE = 10_000

df = pd.read_csv("../jigsaw/dataset_text_target.csv")
print("Size of data:", len(df))

df_true = df[df.target > 0.5]
df_false = df[df.target <= 0.5]
df = pd.concat([df_true[:DATASET_SIZE // 2], df_false[:DATASET_SIZE // 2]], axis=0)
mapper = lambda x: 1 if x > 0.5 else 0
df.target = df.target.apply(mapper)

from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.comment_text, df.target, test_size=0.2, random_state=42,
                                                    shuffle=True)

# Tokenize and encode the data using the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

X_train_encoded = tokenizer.batch_encode_plus(
    x_train.tolist(),
    padding='max_length',
    truncation=True,
    max_length=maximum_length,
    add_special_tokens=True,
    return_tensors='tf',
)
X_test_encoded = tokenizer.batch_encode_plus(
    x_test.tolist(),
    padding='max_length',
    truncation=True,
    max_length=maximum_length,
    add_special_tokens=True,
    return_tensors='tf'
)

# Building the model
model = Sequential()
# Adding the embedding layer to convert input sequences to dense vectors
model.add(Embedding(maximum_features, word_embedding_dims, ))

# Adding the first 1D convolutional layer with ReLU activation
model.add(Conv1D(no_of_filters, kernel_size, padding='valid',
                 activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=3))

model.add(Conv1D(no_of_filters, kernel_size, padding='valid',
                 activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=3))

model.add(Conv1D(no_of_filters, kernel_size, padding='valid',
                 activation='relu', strides=1))
model.add(GlobalMaxPooling1D())

# Adding the dense hidden layer with ReLU activation
model.add(Dense(hidden_dim_1, activation='relu'))

# Adding the output layer with sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

from keras.optimizers import AdamW

# Compiling the model with binary cross-entropy loss and Adam optimizer
model.compile(loss='binary_crossentropy',
              optimizer=AdamW(), metrics=['accuracy'])

# Training the model
history = model.fit(X_train_encoded["input_ids"], y_train, batch_size=batch_size,
                    epochs=epochs, validation_data=(X_test_encoded["input_ids"], y_test))

# Predicting the probabilities for test data
y_pred_prob = model.predict(X_test_encoded["input_ids"])

# Converting the probabilities to binary classes based on a threshold
y_pred = (y_pred_prob > threshold).astype(int)

# Calculating the evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Printing the evaluation metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)

print(classification_report(y_test, y_pred))

import json

with open("result.json", "w") as f:
    json.dump(history.history, f, indent=2)

model.save_pretrained("cnn_model_trained")
