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
            modalities = ['S2L2A']
        )

        self.select_indices = necks.SelectIndices(
            channel_list=[768, 768, 768, 768, 768],
            indices=[3, 5, 7, 9, 11]
        )

        self.reshape_tokens = necks.ReshapeTokensToImage(
            channel_list=[768, 768, 768, 768, 768],
            remove_cls_token=False
        )

        # Project the representations to fit as the skip layers which will connect to 
        self.projections = nn.ModuleList([
            nn.Conv2d(768, 64, kernel_size=1), #index 3
            nn.Conv2d(768, 128, kernel_size=1), #index 5
            nn.Conv2d(768, 256, kernel_size=1), #index 7
            nn.Conv2d(768, 512, kernel_size=1), #index 9
            nn.Conv2d(768, 1024, kernel_size=1), #index 11
        ])

        # Per-level upsampling
        self.upsamplers = nn.ModuleList([
            nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False),  #index 3
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False),  #index 5
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),  #index 7
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  #index 9
            nn.Identity(),  #index 11
        ])


    def forward(self, x):
        embeddings = self.model(x)
        features = self.select_indices(embeddings)      # list of 4 tensors
        features = self.reshape_tokens(features)           # reshape each to (B, C, H, W)

        out = []
        for f, proj, up in zip(features, self.projections, self.upsamplers):
            f = proj(f) 
            f = up(f)
            out.append(f)
        
        
        return out 
