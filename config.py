VERSION = 'version1'

# common config
DATA_DIR = '../../datasets/deepfashion2'
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
CLASSES = ['background',
           'short_sleeved_shirt',
           'long_sleeved_shirt',
           'short_sleeved_outwear',
           'long_sleeved_outwear',
           'vest',
           'sling',
           'shorts',
           'trousers',
           'skirt',
           'short_sleeved_dress',
           'long_sleeved_dress',
           'vest_dress',
           'sling_dress']

SEG_LR = 1e-4


# Triplet model config
PRETRAINED = True
FREEZE = True
OUTPUT_DIM = 64

# Triplet train config
MARGIN = 1.
