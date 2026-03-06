import pandas as pd
from datasets import load_dataset


def build_dataset():

    print("Loading Civil Comments...")

    civil = load_dataset("civil_comments", split="train[:200000]")
    civil_df = pd.DataFrame(civil)

    safe_civil = civil_df[civil_df["toxicity"] < 0.1][["text"]]
    toxic_civil = civil_df[civil_df["toxicity"] > 0.7][["text"]]

    safe_civil["label"] = 0
    toxic_civil["label"] = 1


    print("Loading TweetEval Hate...")

    tweet = load_dataset("tweet_eval", "hate")

    tweet_df = pd.DataFrame(tweet["train"])

    safe_tweet = tweet_df[tweet_df["label"] == 0][["text"]]
    toxic_tweet = tweet_df[tweet_df["label"] == 1][["text"]]

    safe_tweet["label"] = 0
    toxic_tweet["label"] = 1


    print("Combining datasets...")

    safe = pd.concat([safe_civil, safe_tweet])
    toxic = pd.concat([toxic_civil, toxic_tweet])


    print("Balancing dataset...")

    min_size = min(len(safe), len(toxic))

    safe = safe.sample(min_size, random_state=42)
    toxic = toxic.sample(min_size, random_state=42)

    dataset = pd.concat([safe, toxic])

    dataset = dataset.sample(frac=1).reset_index(drop=True)

    dataset.to_csv("data/dataset.csv", index=False)

    print("Dataset saved to data/dataset.csv")


if __name__ == "__main__":
    build_dataset()