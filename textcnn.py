import torch
from torch import nn


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, out_channels):
        super(TextCNN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 100, (n, embedding_dim)) for n in (1, 2, 3)]
        )
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(in_features=300, out_features=out_channels, bias=True)

    def forward(self, x):
        embedded = self.embeddings(torch.transpose(x, 1, 0))
        embedded = embedded.unsqueeze(1)
        convolved = [
            nn.functional.relu(conv(embedded)).squeeze(3)
            for conv in self.convs
            if embedded.size(2) >= conv.kernel_size[0]
        ]
        convolved.extend((3 - len(convolved)) * [convolved[-1]])
        pooled = [
            nn.functional.max_pool1d(convd, int(convd.size(2))).squeeze(2)
            for convd in convolved
        ]
        concatted = torch.cat(pooled, 1)
        dropped = self.dropout(concatted)
        output = self.linear(dropped)
        return output


if __name__ == "__main__":
    pass
