import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


def build_model(num_classes=3):
    # 사전학습된 DeepLabV3 (ResNet50 백본) 불러오기
    model = deeplabv3_resnet50(weights='DEFAULT')

    # 출력 층을 우리 클래스 수(3)에 맞게 교체
    # 기존 마지막 conv를 같은 구조에 출력 채널만 num_classes로 바꾼다
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    # 보조 출력(aux)도 함께 교체 (학습 안정성용, DeepLabV3 기본 구성)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    return model


if __name__ == '__main__':
    # 모델이 제대로 만들어지고 GPU에서 도는지 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(num_classes=3).to(device)

    # 가짜 입력으로 순전파 테스트: 배치2, 채널3, 256x512
    dummy = torch.randn(2, 3, 256, 512).to(device)
    output = model(dummy)

    # DeepLabV3 출력은 딕셔너리, 'out' 키에 예측이 들어있다
    print("출력 타입:", type(output))
    print("out shape:", output['out'].shape)