import torch
import torch.nn as nn
from Embedding import EmbeddingLayer


class MLMAN(nn.Module):
    def __init__(
        self, vocab_size, max_length, word_embedding_dim=50, pos_embedding_dim=5
    ):
        super(MLMAN, self).__init__()
        self.embedding_dim = word_embedding_dim + 2 * pos_embedding_dim
        self.embedding = EmbeddingLayer(
            vocab_size, word_embedding_dim, max_length, pos_embedding_dim
        )

    def context_encoder(self, input):
        input_mask = (input["mask"] != 0).float()
        max_length = input_mask.long().sum(1).max().item()

