# patch embedding
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size, f"Input image size ({H}*{W}) doesn't match model image size ({self.image_size}*{self.image_size})"

        x = self.projection(x).flatten(2).transpose(1, 2)   # [B, embed_dim, num_patches]
        
        return x
    
if __name__ == "__main__":
    image_size = 224
    patch_size = 16
    in_channels = 3
    embed_dim = 768
    x = torch.randn(1, 3, 224, 224)
    model = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
    print(model(x).shape)  # torch.Size([1, 196, 768])