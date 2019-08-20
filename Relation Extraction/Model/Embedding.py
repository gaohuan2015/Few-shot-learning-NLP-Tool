import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_length, position_embedding_dim):
        super(EmbeddingLayer, self).__init__()
        self.wordembedding = nn.Embedding(vocab_size, embedding_dim)
        self.position1_embedding = nn.Embedding(2 * max_length, position_embedding_dim)
        self.position2_embedding = nn.Embedding(2 * max_length, position_embedding_dim)

    def forward(self, inputs):
        word = inputs.word
        pos1 = inputs.pos1
        pos2 = inputs.pos2
        x = torch.cat(
            [
                self.word_embedding(word),
                self.pos1_embedding(pos1),
                self.pos2_embedding(pos2),
            ],
            2,
        )
        return x
