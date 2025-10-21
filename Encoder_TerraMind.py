import torch.nn as nn
from terratorch.registry import BACKBONE_REGISTRY
from terratorch.models import necks
# import albumentations

class TerraMindEncoder(nn.Module):
    def __init__(self,
                 version: str = "terramind_v1_base",
                 pretrained: bool = True,
                 modalities: list[str] = ['S2L2A']
                 ):
        super().__init__()
        self.model = BACKBONE_REGISTRY.build(
            version, 
            pretrained = pretrained,
            modalities = modalities)


    def forward(self, x):
        embeddings = self.model(x)        
        
        return embeddings


