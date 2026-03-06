import pandas as pd

from transformers import AutoTokenizer

from sklearn.model_selection import train_test_split

from src.config import *
from src.utils import set_seed
from src.dataset import PromptDataset
from src.model import load_model
from src.train import train_model
from src.evaluate import evaluate_model


def main():

    set_seed(SEED)

    df = pd.read_csv("data/dataset.csv")

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        stratify=df["label"]
    )

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=0.5,
        stratify=temp_labels
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = PromptDataset(
        train_texts.tolist(),
        train_labels.tolist(),
        tokenizer,
        MAX_LEN
    )

    val_dataset = PromptDataset(
        val_texts.tolist(),
        val_labels.tolist(),
        tokenizer,
        MAX_LEN
    )

    test_dataset = PromptDataset(
        test_texts.tolist(),
        test_labels.tolist(),
        tokenizer,
        MAX_LEN
    )

    model = load_model(MODEL_NAME)

    trainer = train_model(model, train_dataset, val_dataset)

    evaluate_model(trainer, test_dataset)


if __name__ == "__main__":
    main()