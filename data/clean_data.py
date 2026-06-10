import pandas as pd


def clean_file(input_file, output_file):

    df = pd.read_csv(
        input_file,
        sep="\t",
        header=None,
        names=["text", "label"]
    )

    print(f"\nBefore: {len(df)}")

    # remove #NAME?
    df = df[
        ~df["text"].astype(str).str.contains(
            "#NAME?",
            regex=False,
            na=False
        )
    ]

    # remove duplicates
    df = df.drop_duplicates()

    # length
    df["length"] = (
        df["text"]
        .astype(str)
        .str.split()
        .str.len()
    )

    # remove very long texts
    df = df[
        df["length"] <= 100
    ]

    df = df.drop(
        columns=["length"]
    )

    print(f"After: {len(df)}")

    df.to_csv(
        output_file,
        sep="\t",
        header=False,
        index=False,
        encoding="utf-8"
    )


clean_file(
    "data/train.tsv",
    "data/cleaned_train.tsv"
)
