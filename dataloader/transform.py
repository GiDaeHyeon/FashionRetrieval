import albumentationsdef segmentation_transform(mode='train'):    default = [        albumentations.pytorch.ToTensorV2(),        albumentations.Resize(512, 512, p=1),    ]    if mode == 'train':        for i in range(2):            default.append(albumentations.OneOf([                                                albumentations.HorizontalFlip(p=1),                                                albumentations.RandomRotate90(p=1),                                                albumentations.VerticalFlip(p=1)                                            ], p=1))    elif mode == 'val' or mode == 'test':        pass    else:        raise Exception('transform의 mode는 train, val, test 셋 중 하나로 설정해야 합니다.')    transform = albumentations.Compose(default)    return transformdef triplet_transform(mode='train'):    triplet_transform = albumentations.Compose([    # TODO : Triplet Transform 정의    ])