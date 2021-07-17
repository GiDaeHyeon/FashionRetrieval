import albumentations


VERSION = 'version1'

# common config
DATA_DIR = 'data dir'
CKPT_DIR = 'checkpoint dir'
CKPT_NAME = 'checkpoint filename'
BATCH_SIZE = 32
NUM_WORKERS = 8
MAX_EPOCH = 50
GPUS = [0, 1, 2, 3]

# segmentation model config
SEG_ENCODER = 'resnet101'
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
                                    albumentations.pytorch.ToTensor()
                                    ])


# Triplet model config
PRETRAINED = True
FREEZE = True
OUTPUT_DIM = 64

# Triplet train config
MARGIN = 1.

