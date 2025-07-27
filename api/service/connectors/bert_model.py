import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification


# Tokenize and encode the data using the BERT tokenizer

class BertModel:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.MAX_LENGTH = 128
        self.model = TFBertForSequenceClassification.from_pretrained('bert_model_trained', num_labels=2)
        print("Model Loaded")

    def _encode_input(self, text: str):
        encoded = self.tokenizer.batch_encode_plus(
            [text],
            padding='max_length',
            truncation=True,
            max_length=self.MAX_LENGTH,
            add_special_tokens=True,
            return_tensors='tf'
        )
        return encoded

    def predict(self, text: str):
        encoded = self._encode_input(text)
        pred = self.model.predict(
            [encoded['input_ids'], encoded['token_type_ids'], encoded['attention_mask']])
        pred = tf.argmax(pred.logits, axis=1).numpy()
        print(pred)
        return pred[0]
