from sklearn.metrics import classification_report, roc_auc_score
import torch


def evaluate_model(trainer, test_dataset):

    predictions = trainer.predict(test_dataset)

    logits = predictions.predictions
    labels = predictions.label_ids

    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()

    auc = roc_auc_score(labels, probs)

    print("AUC ROC:", auc)

    print(classification_report(labels, logits.argmax(axis=1)))