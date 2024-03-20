#encoder block
import torch
import torch.nn as nn
from Multiheadattn import MultiHeadAttention
from mlp1 import Mlp

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super(EncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=hidden_dim, act_layer=nn.GELU, drop=drop)

    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
if __name__ == "__main__":
    embed_dim = 768
    num_heads = 12
    x = torch.randn(1, 196, 768)
    model = EncoderBlock(embed_dim, num_heads)
    print(model(x).shape)  # torch.Size([1, 196, 768])