import pandas as pd
import random
import re

random.seed(42)


def random_delete(words, p=0.1):
    return [w for w in words if random.random() > p]


def random_swap(words):
    if len(words) < 4:
        return words

    i, j = random.sample(range(len(words)), 2)
    words[i], words[j] = words[j], words[i]
    return words


def repeat_char(text, p=0.2):
    new_text = []

    for char in text:
        new_text.append(char)

        if char.isalpha() and random.random() < p:
            new_text.append(char)

    return "".join(new_text)


def augment_text(text):

    text = repeat_char(text, p=0.15)

    words = text.split()

    if random.random() < 0.5:
        words = random_delete(words, p=0.1)

    if random.random() < 0.3:
        words = random_swap(words)

    return " ".join(words)


def augment_dataframe(df, target_factor=1.2):
    augmented = []

    for _, row in df.iterrows():

        text = str(row["text"])
        sentiment = row["sentiment"]

        if sentiment in ["positive", "neutral"]:

            aug_text = augment_text(text)

            if aug_text != text:
                augmented.append({
                    "text": aug_text,
                    "sentiment": sentiment
                })

    aug_df = pd.DataFrame(augmented)

    return pd.concat([df, aug_df], ignore_index=True)
