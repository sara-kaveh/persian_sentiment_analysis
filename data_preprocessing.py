import pandas as pd
import numpy as np
import joblib
from hazm import Normalizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from config import Config


class DataPreprocessor:

    def __init__(self):
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        self.normalizer = Normalizer()

    def load_data(self, file_path):

        df = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            names=["text", "sentiment"],
            encoding="utf-8"
        )

        df.dropna(inplace=True)
        return df

    def map_sentiment(self, label):

        if label in ["HAPPY", "SURPRISE"]:
            return "positive"

        elif label in ["SAD", "ANGRY", "HATE", "FEAR"]:
            return "negative"

        else:
            return "neutral"

    def fit_transform(self, df):

        texts = [
            self.normalizer.normalize(str(text))
            for text in df["text"]
        ]

        labels = df["sentiment"].apply(
            self.map_sentiment
        ).values

        y = self.label_encoder.fit_transform(labels)

        self.tokenizer = Tokenizer(
            num_words=Config.MAX_FEATURES,
            oov_token="<OOV>"
        )

        self.tokenizer.fit_on_texts(texts)

        sequences = self.tokenizer.texts_to_sequences(texts)

        x = pad_sequences(
            sequences,
            maxlen=Config.MAX_LENGTH,
            padding="post",
            truncating="post"
        )

        return np.array(x), np.array(y)

    def transform(self, df):

        texts = [
            self.normalizer.normalize(str(text))
            for text in df["text"]
        ]

        labels = df["sentiment"].apply(
            self.map_sentiment
        ).values

        y = self.label_encoder.transform(labels)

        sequences = self.tokenizer.texts_to_sequences(texts)

        x = pad_sequences(
            sequences,
            maxlen=Config.MAX_LENGTH,
            padding="post",
            truncating="post"
        )

        return np.array(x), np.array(y)

    def save_preprocessors(self):

        joblib.dump(
            self.tokenizer,
            Config.TOKENIZER_PATH
        )

        joblib.dump(
            self.label_encoder,
            Config.LABEL_ENCODER_PATH
        )

    def load_preprocessors(self):

        self.tokenizer = joblib.load(
            Config.TOKENIZER_PATH
        )

        self.label_encoder = joblib.load(
            Config.LABEL_ENCODER_PATH
        )
