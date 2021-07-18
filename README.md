# FashionRetrieval
**의류 이미지 검색**

이미지 검색 기능을 통해 의류를 검색하는 것은 한 이미지 내 여러 의류가 등장할 수 있다는 점때문에 단순한 이미지 분류로는 수행하기 어렵다.

여기서는 Image Segmentation을 통해 이미지 내 의류 데이터를 추출하고, Metric Learning을 통해 이미지 검색 기능을 구현해보도록 한다.

## Data
Deepfashion2 데이터셋을 활용했다.

(https://github.com/switchablenorms/DeepFashion2)

## Requirements
```
pytorch
torchvision
pytorch-lightning
segmentation-models-pytorch
```
자세한 사항은 레포지토리 내 requirements.txt 참고바람.
