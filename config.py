import albumentations
from albumentations.pytorch import ToTensorV2


VERSION = 'version1'

# common config
DATA_DIR = './datasets/'
CKPT_DIR = './checkpoints/segmentation/'
CKPT_NAME = f'{VERSION}_checkpoint'
BATCH_SIZE = 32
NUM_WORKERS = 8
MAX_EPOCH = 50
GPUS = [0, 1, 2, 3]

# segmentation model config
SEG_ENCODER = 'resnet50'
SEG_ENCODER_DEPTH = 5
SEG_ENCODER_WEIGHT = 'imagenet'
CLASS_NUM = 14

# segmentation train config
TRANSFORM = albumentations.Compose([
                                    albumentations.Resize(256, 256),
                                    albumentations.RandomCrop(224, 224),
                                    albumentations.OneOf([
                                                          albumentations.HorizontalFlip(p=1),
                                                          albumentations.RandomRotate90(p=1),
                                                          albumentations.VerticalFlip(p=1)
                                                         ], p=1),
                                    albumentations.OneOf([
                                                          albumentations.HorizontalFlip(p=1),
                                                          albumentations.RandomRotate90(p=1),
                                                          albumentations.VerticalFlip(p=1)
                                                         ], p=1),
                                    ToTensorV2(),
                                    ])


# Triplet model config
PRETRAINED = True
FREEZE = True
OUTPUT_DIM = 64

# Triplet train config
MARGIN = 1.

