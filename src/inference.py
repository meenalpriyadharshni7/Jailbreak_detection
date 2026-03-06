import torch


def predict(prompt, model, tokenizer, max_len):

    encoding = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(**encoding)

        probs = torch.softmax(outputs.logits, dim=1)

        prediction = torch.argmax(probs).item()

    return prediction, probs.tolist()