from torchtext.data import TabularDataset, Field
import jieba_fast
import json
import pandas as pd


def get_dataset(max_length):
    sentence = Field(
        include_lengths=True,
        fix_length=max_length,
        tokenize=jieba_fast.lcut,
    )
    label = Field(tokenize=lambda x: x.split("|"), unk_token=None)
    for file in ["train", "test"]:
        records = []
        with open(f"./百度事件抽取/multi-classification-{file}.txt", "r") as f:
            for line in f.readlines():
                l, s = line.split(" ")
                records.append({"label": l, "sentence": s.replace("\n", "")})
        df = pd.DataFrame(records)
        df = df[["sentence", "label"]]
        df.to_csv(f"./百度事件抽取/{file}.tsv", sep="\t", index=False)
    train = TabularDataset(
        path="./百度事件抽取/train.tsv",
        format="tsv",
        skip_header=True,
        fields=[("sentence", sentence), ("label", label)],
    )
    dev = TabularDataset(
        path="./百度事件抽取/test.tsv",
        format="tsv",
        skip_header=True,
        fields=[("sentence", sentence), ("label", label)],
    )
    sentence.build_vocab(train)
    label.build_vocab(train)

    return sentence, label, train, dev


if __name__ == "__main__":
    sentence, label, train, dev = get_dataset(400)
