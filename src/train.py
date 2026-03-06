from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np


def compute_metrics(eval_pred):

    logits, labels = eval_pred

    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary"
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def train_model(model, train_dataset, val_dataset):

    training_args = TrainingArguments(

        output_dir="./results",

        learning_rate=2e-5,

        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,

        num_train_epochs=5,

        evaluation_strategy="epoch",

        save_strategy="epoch",

        load_best_model_at_end=True,

        fp16=True,

        logging_dir="./logs"
    )

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=val_dataset,

        compute_metrics=compute_metrics
    )

    trainer.train()

    return trainer