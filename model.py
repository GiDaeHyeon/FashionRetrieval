import segmentation_models_pytorch as smp

import torch.nn as nn
from torchvision.models import resnext50_32x4d

from config import *


class SegmentationModel(nn.Module):
    def __init__(self,
                 # segmentation model config
                 seg_encoder=SEG_ENCODER,
                 seg_encoder_depth=SEG_ENCODER_DEPTH,
                 seg_encoder_weight=SEG_ENCODER_WEIGHT,
                 classes_num=CLASS_NUM
                 ):
        super(SegmentationModel, self).__init__()
        self.segmentation_model = smp.DeepLabV3Plus(
            encoder_name=seg_encoder,
            encoder_depth=seg_encoder_depth,
            encoder_weights=seg_encoder_weight,
            classes=classes_num
        )

    def forward(self, x):
        return self.segmentation_model(x)


class TripletModel(nn.Module):
    def __init__(self,
                 # classification model config
                 pretrained=PRETRAINED,
                 freeze=FREEZE,
                 output_dim=OUTPUT_DIM
                 ):
        super(TripletModel, self).__init__()
        self.cnn_model = resnext50_32x4d(pretrained=pretrained)

        if freeze:
            for parameter in self.cnn_model.parameters():
                parameter.requires_grad = False

        in_features = self.cnn_model.fc.in_features

        self.cnn_model.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, output_dim)
        )

    def forward(self, anchor, positive, negative):
        # TODO : 여기에서 augmentation을 수행해줘도 괜찮을까요?
        anchor = self.cnn_model(anchor)
        positive = self.cnn_model(positive)
        negative = self.cnn_model(negative)
        return anchor, positive, negative

    def extract_feature(self, x):
        return self.cnn_model(x)
