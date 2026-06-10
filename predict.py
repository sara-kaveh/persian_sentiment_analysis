import numpy as np
from hazm import Normalizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import Config


class Predictor:

    def __init__(self, model_path, tokenizer, label_encoder):
        self.model = load_model(model_path, compile=False)
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.normalizer = Normalizer()

    def predict_batch(self, texts):

        normalized_texts = [
            self.normalizer.normalize(text)
            for text in texts
        ]

        sequences = self.tokenizer.texts_to_sequences(
            normalized_texts
        )

        padded = pad_sequences(
            sequences,
            maxlen=Config.MAX_LENGTH,
            padding="post",
            truncating="post"
        )

        predictions = self.model.predict(
            padded,
            verbose=0
        )

        results = []

        for text, pred in zip(texts, predictions):

            predicted_class = np.argmax(pred)

            results.append({
                "text": text,
                "sentiment": self.label_encoder.inverse_transform(
                    [predicted_class]
                )[0],
                "confidence": float(np.max(pred))
            })

        return results
