class Config:
    # Dataset
    TRAIN_PATH = "data/cleaned_train.tsv"
    TEST_PATH = "data/test.tsv"

    # Preprocessing
    MAX_FEATURES = 11000
    MAX_LENGTH = 70

    # Model
    EMBEDDING_DIM = 256
    LSTM_UNITS = 64

    # Training
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001

    # Saved files
    MODEL_PATH = "models/best_model.keras"

    TOKENIZER_PATH = "models/tokenizer.pkl"
    LABEL_ENCODER_PATH = "models/label_encoder.pkl"

    RANDOM_SEED = 123
