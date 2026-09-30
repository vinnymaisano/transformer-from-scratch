import numpy as np
from gpt.nn import Sequential, Linear, LayerNorm, ReLU
from gpt.transformer import TokenAndPositionEmbedding, Block

# builds a GPT model by prepending token & position embeddings to a chain of transformer blocks, then

# passing output through a final linear layer
class GPT(Sequential):
    def __init__(self, vocab_size, block_size, d_model, num_heads, num_layers,
                 dropout=0.0, pre_norm=True, act=ReLU):
        self.block_size = block_size
        super().__init__(
            TokenAndPositionEmbedding(vocab_size, block_size, d_model),
            *[Block(d_model, num_heads, dropout, pre_norm, act) for _ in range(num_layers)],
            LayerNorm(d_model),
            Linear(d_model, vocab_size),
        )