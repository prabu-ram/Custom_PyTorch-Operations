import torch
from torch import nn
from torchvision.ops import roi_pool
from compiler import *

assert torch.backends.mps.is_available()
mps_device = torch.device("mps")  # Device object representing GPU.

# Wrapper over the custom MPS ROI Pooling kernel.
class MPSROIPooling(nn.Module):
    __constants__ = ["pool_height", "pool_width"]
    pool_height: int
    pool_width: int

    def __init__(self, pool_height: int = 7, pool_width: int = 7) -> None:
        super().__init__()
        self.pool_height = pool_height
        self.pool_width = pool_width

    def forward(self, feature_map, rois):
        return compiled_lib.mps_roipooling(feature_map, rois, self.pool_height, self.pool_width)

    def extra_repr(self):
        return f'pool_height={self.pool_height}, pool_width={self.pool_width}'

# Wrapper over a model using the custom MPS ROI Pooling implementation.
class CustomMPSROIPoolingModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        pool_height: int = 7,
        pool_width: int = 7
    ):
        super().__init__()
        self.backbone = nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1)  # Example backbone
        self.roi_pooling = MPSROIPooling(pool_height=pool_height, pool_width=pool_width)
        self.fc = nn.Linear(512 * pool_height * pool_width, num_classes)

    def forward(self, x, rois):
        feature_map = self.backbone(x)
        pooled_features = self.roi_pooling(feature_map, rois)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        #print(self.fc(pooled_features))
        return self.fc(pooled_features)

# Wrapper over a model using the default PyTorch ROI Pooling implementation.
class TorchvisionROIPoolingModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        pool_height: int = 7,
        pool_width: int = 7,
        spatial_scale: float = 1.0
    ):
        super().__init__()
        self.backbone = nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1)  # Example backbone
        self.roi_pooling = roi_pool
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.spatial_scale = spatial_scale
        self.fc = nn.Linear(512 * pool_height * pool_width, num_classes)

    def forward(self, x, rois):
        feature_map = self.backbone(x)
        pooled_features = self.roi_pooling(feature_map, rois, output_size=(self.pool_height, self.pool_width), spatial_scale=self.spatial_scale)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        return self.fc(pooled_features)
