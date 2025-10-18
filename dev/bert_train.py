import pandas as pd

data_size = 1_000
epochs = 10

df = pd.read_csv("../jigsaw/dataset_text_target.csv").dropna()
# df_true = df[df.target > 0.5]
# df_false = df[df.target <= 0.5]

# df = pd.concat([df_true[:data_size // 2], df_false[:data_size // 2]], axis=0)

mapper = lambda x: 1 if x > 0.7 else 0
df.target = df.target.apply(mapper)

from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.comment_text, df.target, test_size=0.2, random_state=42,
                                                    stratify=df.target, shuffle=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

max_len = 128

# In[12]:


X_train_encoded = tokenizer.batch_encode_plus(
    x_train.tolist(),
    padding='max_length',
    truncation=True,
    max_length=max_len,
    add_special_tokens=True,
    return_tensors='tf'
)
X_test_encoded = tokenizer.batch_encode_plus(
    x_test.tolist(),
    padding='max_length',
    truncation=True,
    max_length=max_len,
    add_special_tokens=True,
    return_tensors='tf'
)

# In[13]:


model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# In[14]:


import tensorflow as tf

# In[15]:


optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# In[16]:


history = model.fit(
    [X_train_encoded['input_ids'], X_train_encoded['token_type_ids'], X_train_encoded['attention_mask']],
    y_train,
    validation_data=(
        [X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']], y_test),
    batch_size=64,
    epochs=epochs
)

from sklearn.metrics import classification_report

# In[19]:


y_pred = model.predict(
    [X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']])
y_pred = tf.argmax(y_pred.logits, axis=1).numpy()

# In[20]:


print(classification_report(y_test, y_pred))

import json

with open("result.json", "w") as f:
    json.dump(history.history, f, indent=2)

# Save history (loss and accuracy for train and validation) to CSV as well
hist_df = pd.DataFrame(history.history)
hist_df.index = hist_df.index + 1  # start epochs at 1 for readability
hist_df.to_csv("history.csv", index_label="epoch")

model.save_pretrained("bert_model_trained")

# In[ ]:
