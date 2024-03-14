import torch
import torch.nn as nn
import torchvision

class ViT_16_mod(nn.Module):
    def __init__(self,n_classes,dropout=.4):
        super(ViT_16_mod, self).__init__()
        self.model_original = torchvision.models.vit_b_16(pretrained=True)

        self._process_input = self.model_original._process_input
        self.encoder = self.model_original.encoder
        self.heads = nn.Sequential(nn.Dropout(dropout), nn.Linear(768, n_classes))
        self.class_token = self.model_original.class_token

    def forward_encoder(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        return x
    
    def forward(self,x:torch.Tensor,epoch=10000000):
        #x = self.forward_encoder(x)
        if epoch>5:
            x = self.forward_encoder(x)
        else:
            with torch.no_grad():
                x = self.forward_encoder(x)
        x = self.heads(x)
        return x