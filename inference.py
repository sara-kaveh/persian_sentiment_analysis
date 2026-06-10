import tkinter as tk
from tkinter import ttk
from config import Config
from data_preprocessing import DataPreprocessor
from predict import Predictor


class EmotionClassifierApp:

    def __init__(self, root):

        self.root = root
        self.root.title("Persian Emotion Classifier")
        self.root.geometry("900x550")
        self.root.resizable(False, False)

        preprocessor = DataPreprocessor()
        preprocessor.load_preprocessors()

        self.predictor = Predictor(
            Config.MODEL_PATH,
            preprocessor.tokenizer,
            preprocessor.label_encoder
        )

        title = ttk.Label(
            root,
            text="Persian Emotion Classifier",
            font=("Arial", 24, "bold")
        )
        title.pack(pady=15)

        self.text_input = tk.Text(
            root,
            font=("Arial", 19),
            height=5,
            width=60
        )
        self.text_input.pack(pady=10)

        predict_button = ttk.Button(
            root,
            text="Predict",
            command=self.predict
        )
        predict_button.pack(pady=10)

        self.result_label = ttk.Label(
            root,
            text="Prediction will appear here",
            font=("Arial", 20)
        )
        self.result_label.pack(pady=10)

        self.confidence_label = ttk.Label(
            root,
            text="",
            font=("Arial", 20)
        )
        self.confidence_label.pack()

    def predict(self):

        text = self.text_input.get(
            "1.0",
            tk.END
        ).strip()

        if not text:
            self.result_label.config(
                text="Please enter some text."
            )
            self.confidence_label.config(text="")
            return

        result = self.predictor.predict_batch(
            [text]
        )[0]

        self.result_label.config(
            text=f"Sentiment: {result['sentiment']}"
        )

        self.confidence_label.config(
            text=f"Confidence: {result['confidence']:.2%}"
        )


def main():

    root = tk.Tk()
    app = EmotionClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
