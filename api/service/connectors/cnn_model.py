from keras.models import load_model
from transformers import BertTokenizer


class CNNModel:

    def __init__(self):
        self.model = load_model("cnn_model_trained/cnn_model_trained.keras")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.maximum_length = 128  # Maximum length of input sequences
        self.threshold = 0.5

    def encode(self, text):
        return self.tokenizer.batch_encode_plus(
            [text],
            padding='max_length',
            truncation=True,
            max_length=self.maximum_length,
            add_special_tokens=True,
            return_tensors='tf'
        )

    def predict(self, text):
        encoded = self.encode(text)
        y_pred_prob = self.model.predict(encoded["input_ids"])
        y_pred = (y_pred_prob > self.threshold).astype(int)
        return y_pred[0]
