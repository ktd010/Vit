# mlp head for outside

import torch
import torch.nn as nn

class MlpHead(nn.Module):
    def __init__(self, in_features, hidden_features=3000, out_features=1000, act_layer=nn.GELU, drop=0.):
        super(MlpHead, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
    
if __name__ == "__main__":
    in_features = 768
    hidden_features = 3072
    out_features = 1000
    x = torch.randn(1, 196, 768)
    model = MlpHead(in_features, hidden_features, out_features)
    print(model(x).shape)  # torch.Size([1, 196, 1000])