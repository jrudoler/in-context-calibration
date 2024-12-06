import torch
import lightning as L
from torch.nn.functional import scaled_dot_product_attention


class ToyTransformer(L.LightningModule):
    def __init__(self, embed_dim: int = 10, output_dim: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        self.embedding = torch.nn.Embedding(embed_dim, embed_dim)
