import torchvision.models as models
import torch.nn as nn


def build_model(pretrained=True, fine_tune=True, num_classes=10):
    if pretrained:
        print("[INFO]: Loading pre-trained weights")
    else:
        print("[INFO]: Not loading pre-trained weights")
    model = models.densenet121(weights=pretrained)

    if fine_tune:
        print("[INFO]: Fine-tuning all layers...")
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print("[INFO]: Freezing hidden layers...")
        for params in model.parameters():
            params.requires_grad = False

    # Change the final classification head.
    # for effnetv2
    # model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    model.classifier[1] = nn.Linear(num_features=64, num_classes=5)
    return model
