# segmentation model config
SEG_ENCODER = 'resnet101'
SEG_ENCODER_DEPTH = 5
SEG_ENCODER_WEIGHT = 'imagenet'
CLASS_NUM = 14

# classification model config
PRETRAINED = True
FREEZE = True
OUTPUT_DIM = 64

# train config
NUM_PROCESSES = 4
MAX_EPOCH = 30
