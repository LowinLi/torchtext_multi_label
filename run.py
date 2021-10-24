from torchtext.vocab import Vectors
from torchtext.data import BucketIterator
from torch import nn, optim
import torch
import random
import pandas as pd
import jieba_fast
from tqdm import tqdm
import os
import numpy as np
import json
import time
import onnxruntime
from get_dataset import get_dataset
from textcnn import TextCNN

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
root = os.path.dirname(os.path.abspath(__file__))
print(root)


class ModelLabel:
    def __init__(
        self, dim=300, device="cpu", batch_size=200, lr=5e-3, epochs=5, max_length=100
    ):
        self.dim = dim
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.max_length = max_length

    def train(self, pretrained_file):
        sentence, label, train, dev = get_dataset(self.max_length)
        sentence.vocab.load_vectors(
            vectors=Vectors(os.path.join(root, pretrained_file))
        )
        self.model = TextCNN(
            sentence.vocab.vectors.shape[0],  # 词个数
            sentence.vocab.vectors.shape[1],  # 向量维度
            len(label.vocab.stoi),  # 标签个数
        )
        self.model.embeddings.weight.data.copy_(sentence.vocab.vectors)
        datas = (train, dev)
        train_iter, val_iter = BucketIterator.splits(
            datas,
            device=self.device,
            repeat=False,
            batch_size=self.batch_size,
            sort=False,
            shuffle=True,
        )
        self.model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        epochs = []
        train_accs = []
        val_accs = []
        acc, loss = self.validate_multi_label(val_iter)
        print(f"验证集准确率:\t{acc}\t\tloss:\t{loss}")
        bars = tqdm(range(1, self.epochs + 1))
        for epoch in bars:
            epochs.append(epoch)
            self.model.train()

            train_epoch_loss = 0
            total_batch = len(train_iter)
            for batch, batch_data in enumerate(train_iter):
                bars.set_description(f"batch{batch}, total{total_batch}")
                x = batch_data.sentence
                y = batch_data.label
                optimizer.zero_grad()
                proba = self.model.forward(x[0])
                gold_label_metric = self.multi_label_metrics_transfer(
                    y, len(label.vocab.stoi)
                )
                loss = self.criterion(proba[:, 1:], gold_label_metric)
                loss.backward()
                optimizer.step()
                train_epoch_loss += float(loss.data)
            acc, loss = self.validate_multi_label(train_iter)
            print(f"训练集准确率:\t{acc}\t\tloss:\t{loss}")
            train_accs.append(round(acc, 3))
            acc, loss = self.validate_multi_label(val_iter)
            print(f"验证集准确率:\t{acc}\t\tloss:\t{loss}")
            val_accs.append(round(acc, 3))

        with open("metric.md", "w") as f:
            f.write("## 这是一个CML报告，在ACTIONS中跑的textcnn模型训练和评测结果\n\n")
            f.write("---\n")
            f.write("|epoch|训练集loss|验证集loss|训练集acc|验证集acc|\n")
            f.write("|-|-|-|-|-|\n")
            for epoch, train_loss, val_loss, train_acc, val_acc in zip(
                epochs, train_epoch_loss_list, val_epoch_loss_list, train_accs, val_accs
            ):
                f.write(f"|{epoch}|{train_loss}|{val_loss}|{train_acc}|{val_acc}|\n")

        # 存onnx格式
        dummy = torch.zeros(self.max_length, 1).long()
        torch.onnx.export(
            self.model.to("cpu"),
            dummy,
            f="model.onnx",
            export_params=True,
            verbose=False,
            opset_version=12,
            training=False,
            do_constant_folding=False,
            input_names=["input"],
            output_names=["output"],
        )
        # 存字典
        with open("vocab.json", "w") as f:
            json.dump(dict(sentence.vocab.stoi), f, indent=4, ensure_ascii=False)
        # 存标签
        with open("reverse_label.json", "w") as f:
            label_dict = label.vocab.stoi
            reverse_label_dict = dict(zip(label_dict.values(), label_dict.keys()))
            json.dump(reverse_label_dict, f, indent=4, ensure_ascii=False)

    def multi_label_metrics_transfer(self, y, label_num):
        """
        输入 torchtext 的多分类标签体系，0表示占位符没有意义
        tensor([[1, 1, 1, 2, 1, 1, 1, 1, 1],
                [2, 3, 2, 1, 0, 2, 2, 0, 2],
                [0, 0, 3, 0, 0, 0, 0, 0, 0]])
        输出 onehot多标签矩阵
        tensor([[1., 1., 0.],
                [1., 0., 1.],
                [1., 1., 1.],
                [1., 1., 0.],
                [1., 0., 0.],
                [1., 1., 0.],
                [1., 1., 0.],
                [1., 0., 0.],
                [1., 1., 0.]])
        """
        return torch.zeros(
            y.shape[1],
            label_num,
            dtype=torch.float,
            device=self.device,
        ).scatter_(
            1,
            y.T,
            torch.ones(
                y.shape[1],
                label_num,
                dtype=torch.float,
                device=self.device,
            ),
        )[
            :, 1:
        ]

    def load(self):
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = 1
        self.model = onnxruntime.InferenceSession("model.onnx", sess_options)
        with open("vocab.json", "r") as f:
            self.vocab = json.load(f)
        with open("reverse_label.json", "r") as f:
            self.reverse_label = json.load(f)

    def predict(self, sentence):
        # 编码
        inputs = []
        for word in jieba_fast.lcut(sentence):
            if word in self.vocab.keys():
                inputs.append([self.vocab[word]])
            else:
                inputs.append([self.vocab["<unk>"]])

        # 强行padding至最大长度
        if len(inputs) <= self.max_length:
            inputs = inputs + [[self.vocab["<pad>"]]] * (self.max_length - len(inputs))
        else:
            inputs = inputs[: self.max_length]
        # 输入转换格式
        onnx_inputs = {"input": np.array(inputs)}
        # onnx推断
        output = self.model.run(None, onnx_inputs)[0][0]
        probas = 1 / (1 + np.exp(-output))
        labels = []
        scores = []
        for i, proba in enumerate(probas):
            if i > 0 and proba > 0.5:
                labels.append(self.reverse_label[str(i)])
                scores.append(proba)
        return labels, scores

    def validate_multi_label(self, val_iter):
        self.model.eval()
        with torch.no_grad():
            y_p, y_t = [], []
            loss = 0
            for batch in val_iter:
                x, y = batch.sentence, batch.label
                output = self.model.forward(x[0])
                probas = torch.sigmoid(output)
                predict_metrics = torch.where(
                    probas > 0.5,
                    torch.ones(probas.shape[0], probas.shape[1], device=self.device),
                    torch.zeros(probas.shape[0], probas.shape[1], device=self.device),
                )[:, 1:]
                gold_label_metric = self.multi_label_metrics_transfer(
                    y.data, probas.shape[1]
                )
                loss += float(self.criterion(probas[:, 1:], gold_label_metric).data)
                y_p += predict_metrics.tolist()
                y_t += gold_label_metric.tolist()
            correct = sum([1 if i == j else 0 for i, j in zip(y_p, y_t)])
            accuracy = correct / len(y_p)
            loss = loss / len(y_p)
        return accuracy, loss

    def evaluate(self):
        # 批测流程
        self.load()
        df = pd.read_csv("./百度事件抽取/test.tsv", sep="\t")
        start = time.time()
        correct = 0
        for x in df.to_dict(orient="records"):
            labels, scores = self.predict(x["sentence"])
            if "|".join(labels) == x["label"]:
                correct += 1
        waste_time = time.time() - start
        qps = round(len(df) / waste_time, 3)
        accuracy = round(correct / len(df), 3)
        waste_every_record = round(waste_time / len(df) * 1000, 3)
        with open("evaluate.md", "w") as f:
            f.write("\n---\n")
            f.write("+ 预测\n\n")
            f.write("|指标|值|\n")
            f.write("|-|-|\n")
            f.write(f"|单条用时（ms）|{waste_every_record}|\n")
            f.write(f"|单秒执行条数|{qps}|\n")
            f.write(f"|准确率|{accuracy}|\n")
            f.write("---\n")
            f.write("+ 参考\n\t+ [cml官网](https://cml.dev/)\n")
            f.write("---\n")

        return round(accuracy, 3)


if __name__ == "__main__":
    m = ModelLabel(device="cpu", max_length=100, epochs=7)
    m.train("sgns_百度")
    print(m.evaluate())
