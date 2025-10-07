import torch.nn as nn
from terratorch.registry import BACKBONE_REGISTRY
from terratorch.models import necks
import albumentations

class TerraMindEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BACKBONE_REGISTRY.build(
            'terramind_v1_base', 
            pretrained = True,
            modalities = ['S2L2A'])


    def forward(self, x):
        embeddings = self.model(x)        
        
        return embeddings 


