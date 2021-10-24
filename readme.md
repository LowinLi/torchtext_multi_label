## 简要
+ **解读**：torchtext库，做多标签任务
+ **实践**：textcnn模型，跑[百度事件多标签比赛](https://aistudio.baidu.com/aistudio/competition/detail/32/0/introduction)，验证集准确率accuracy达到`86%`
+ **运行**：`github`的`action`中，完成全程训练、批测，结果报告通过`cml工具`发送至commit评论

## 解读

+ 如何用torchtext库，做多标签任务

#### 读取数据

 顾名思义，多标签任务像是`不定项选择题`，是一条样本对应一个或多个标签，也可以没有对应标签，所以标注字段不能再用`sequential=False`参数，要对标注列进行切分，源码中的注释说明：

> sequential: Whether the datatype represents sequential data. If False, no tokenization is applied. Default: True.

##### 写入标签数据
+ `<pad>`表示占位符没有任何意义

```shell
echo -e "label\n汽车&&银行\n汽车&&天气\n天气\n<pad>\n咖啡" > label.tsv
```


##### 字段设置

以`&&`切分标注标签，那么可以这样写：
```python
label = torchtext.data.Field(
    tokenize=lambda x: x.split("&&"), unk_token=None
    )

```

##### 读取Dataset

```python
dataset = torchtext.data.TabularDataset(
            path="label.tsv",
            format="tsv",
            skip_header=True,
            fields=[("label", label)],
        )
print(dataset.examples[0].__dict__)
print(dataset.examples[1].__dict__)
print(dataset.examples[2].__dict__)
print(dataset.examples[3].__dict__)
print(dataset.examples[4].__dict__)
```
>{'label': ['汽车', '银行']}
>{'label': ['汽车', '天气']}
>{'label': ['天气']}
>{'label': ["`<pad>`"]}
>{'label': ["咖啡"]}

##### 建立标签id映射表

```python
label.build_vocab(dataset)
with open("label_dict.json", "w") as f:
    json.dump(
        dict(label.vocab.stoi),
        f,
        indent=4,
        ensure_ascii=False
    )
print(label.vocab.stoi)
```
> {'`<pad>`': 0, '天气': 1, '汽车': 2, '咖啡': 3, '银行': 4}

##### 读取Iterator
```python
train_iter = torchtext.data.BucketIterator(
        dataset,
        device="cpu",
        repeat=False,
        batch_size=2,
        sort=False,
        shuffle=True,
    )
for batch in train_iter:
    print(batch.label)
```

```
# 注意这里每一列是一个样本
tensor([[2],
        [4]])  # 汽车、银行;
tensor([[2, 1],
        [1, 0]]) # 汽车、天气；天气、<pad>
tensor([[3, 0]]) # 咖啡;<pad>
```


#### 损失函数

+ 多分类任务相当于`单选题`,每个标签是`互斥的`，预测整个logits在这个标签体系上做`softmax`，然后计算交叉熵损失函数；
+ 多标签任务相当于`不定项选择题`,每个标签是`独立的`，每个标签位置的logit做`sigmoid`，然后单独计算交叉熵损失函数，然后再求和；
+ pytorch已经包装好多标签任务的损失函数[BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)

```python
torch.nn.BCEWithLogitsLoss
```

#### 标注标签转onehot格式
+ 当前从生成器BucketIterator出来的batch数据，是标签id，需要把它映射成`onehot`格式便于计算`Loss`

```python
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
        device=self.config.device,
    ).scatter_(
        1,
        y.T,
        torch.ones(
            y.shape[1],
            label_num,
            dtype=torch.float,
            device=self.config.device,
        ),
    )[
        :, 1:
    ]
```

## 实践
+ [数据源](https://github.com/percent4/multi-label-classification-4-event-type/tree/master/data)
+ [比赛](https://aistudio.baidu.com/aistudio/competition/detail/32/0/introduction)

#### 统计

| 数据项目                       | 统计值 |
| ------------------------------ | ------ |
| 训练集单条jieba分词个数均值    | 31.63  |
| 训练集单条jieba分词个数98%分位 | 101    |
| 训练集条数                     | 11958  |
| 验证集条数                     | 1498   |
| jieba分词去重个数              | 40113  |
| 标签个数                       | 65     |

因此模型最大长度设置`100`

## 运行

详见本仓库actions

## 参考致谢
1. [CLUE](https://github.com/CLUEbenchmark/CLUE)
2. [cml](https://towardsdatascience.com/what-data-scientists-need-to-know-about-devops-2f8bc6660284?gi=d43983ac072b)
3. [cml.dev](https://cml.dev/)
4. [onnx](https://github.com/microsoft/onnxruntime)
5. [textcnn](http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf)
6. [word2vec](https://github.com/Embedding/Chinese-Word-Vectors)
---
+ 欢迎`Star`和`Fork`
+ 欢迎订阅我的博客[https://lowin.li](https://lowin.li)