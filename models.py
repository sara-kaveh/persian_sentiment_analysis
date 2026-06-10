from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from config import Config


class SentimentModel:

    @staticmethod
    def build(vocab_size, num_classes):

        model = Sequential([

            Embedding(
                vocab_size,
                Config.EMBEDDING_DIM,
                input_length=Config.MAX_LENGTH
            ),

            Bidirectional(
                LSTM(
                    Config.LSTM_UNITS,
                    return_sequences=True,
                    dropout=0.2,
                    recurrent_dropout=0.2
                )
            ),

            Bidirectional(
                LSTM(
                    Config.LSTM_UNITS // 2,
                    dropout=0.2,
                    recurrent_dropout=0.2
                )
            ),

            Dense(32, activation="relu", kernel_regularizer=l2(0.01)),

            Dropout(0.5),

            Dense(
                num_classes,
                activation="softmax"
            )
        ])

        model.compile(
            optimizer=Adam(
                learning_rate=Config.LEARNING_RATE
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model
