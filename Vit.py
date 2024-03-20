# Vit
import torch
import torch.nn as nn
from patchembed import PatchEmbedding
from Encoder import EncoderBlock
from Mlp2 import MlpHead

from torchinfo import summary

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_layers, embed_dim, num_heads,):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, embed_dim)
        self.num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches+1, embed_dim))
        self.encoder = nn.ModuleList([EncoderBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.mlp_head = MlpHead(embed_dim, hidden_features=embed_dim, out_features=num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for encoder_block in self.encoder:
            x = encoder_block(x)
        x = self.mlp_head(x[:, 0])
        return x
    
if __name__ == "__main__":
    image_size = 224
    patch_size = 16
    num_classes = 1000
    num_layers = 12
    embed_dim = 768
    num_heads = 12
    x = torch.randn(10, 3, 224, 224)
    model = VisionTransformer(image_size, patch_size, num_classes, num_layers, embed_dim, num_heads)
    print(model(x).shape)  # torch.Size([1, 1000])


        
#image_path = "/mnt/c/Datasets/MSCoco/val2017/000000000285.jpg"
    
summary(model, input_size=(10, 3, 224, 224))