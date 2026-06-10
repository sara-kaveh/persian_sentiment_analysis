from augmentation import augment_dataframe
from tensorflow.keras.models import load_model
from predict import Predictor
from train import Trainer
from models import SentimentModel
from data_preprocessing import DataPreprocessor
from config import Config
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import numpy as np
import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"


def main():

    np.random.seed(Config.RANDOM_SEED)
    tf.random.set_seed(Config.RANDOM_SEED)

    preprocessor = DataPreprocessor()

    # Load official split
    train_df = preprocessor.load_data(
        Config.TRAIN_PATH
    )

    test_df = preprocessor.load_data(
        Config.TEST_PATH
    )

    all_df = pd.concat(
        [train_df, test_df],
        ignore_index=True
    )

    all_df["stratify_label"] = all_df["sentiment"].apply(
        preprocessor.map_sentiment
    )

    train_df, test_df = train_test_split(
        all_df,
        test_size=0.15,
        random_state=Config.RANDOM_SEED,
        stratify=all_df["stratify_label"]
    )

    train_df = train_df.drop(columns=["stratify_label"])
    test_df = test_df.drop(columns=["stratify_label"])

    train_df["stratify_label"] = train_df["sentiment"].apply(
        preprocessor.map_sentiment
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.15,
        random_state=Config.RANDOM_SEED,
        stratify=train_df["stratify_label"]
    )

    train_df = train_df.drop(columns=["stratify_label"])
    val_df = val_df.drop(columns=["stratify_label"])

    train_df = augment_dataframe(train_df)

    x_train, y_train = preprocessor.fit_transform(
        train_df
    )

    x_val, y_val = preprocessor.transform(
        val_df
    )

    # Transform test
    x_test, y_test = preprocessor.transform(
        test_df
    )

    model = SentimentModel.build(
        vocab_size=min(
            Config.MAX_FEATURES,
            len(preprocessor.tokenizer.word_index) + 1
        ),
        num_classes=len(
            preprocessor.label_encoder.classes_
        )
    )

    trainer = Trainer(model)

    history = trainer.train(
        x_train,
        y_train,
        x_val,
        y_val
    )

    best_model = load_model(Config.MODEL_PATH)

    trainer.model = best_model

    trainer.evaluate(
        x_test,
        y_test,
        preprocessor.label_encoder
    )

    preprocessor.save_preprocessors()

    predictor = Predictor(
        Config.MODEL_PATH,
        preprocessor.tokenizer,
        preprocessor.label_encoder
    )

    samples = [
        "خیلی خوشحال شدم",
        "از این اتفاق عصبانی هستم",
        "واقعا ناراحت شدم",
        "باورم نمیشه!",
        "بد نبود",
        "معمولی بود",
        "عالی بود",
        "زیاد خوب نبود",
        "از این متنفرم",
        "عادی بود"
    ]

    print("\nExample Predictions")
    print("=" * 50)

    results = predictor.predict_batch(samples)

    for result in results:
        print(result)


if __name__ == "__main__":
    main()
