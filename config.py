VERSION = 'version1'

# common config
DATA_DIR = './datasets/deepfashion2/'
CKPT_DIR = './checkpoints/FashionRetrieval/'
CKPT_NAME = f'{VERSION}_checkpoint'
BATCH_SIZE = 32
NUM_WORKERS = 8
MAX_EPOCH = 1000
GPUS = 4

# segmentation model config
SEG_ENCODER = 'resnet50'
SEG_ENCODER_DEPTH = 5
SEG_ENCODER_WEIGHT = 'imagenet'
CLASS_NUM = 14

SEG_LR = 1e-3


# Triplet model config
PRETRAINED = True
FREEZE = True
OUTPUT_DIM = 64

# Triplet train config
MARGIN = 1.

