import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class AbolfazNework(nn.Module):
    def __init__(self):
        super(AbolfazNework, self).__init__()
        self.pretrained_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for layer in self.pretrained_model.layer1.parameters():
            layer.requires_grad = False
        self.pretrained_model.conv1.weight.requires_grad = False
        self.pretrained_model.bn1.weight.requires_grad = False

        num_features = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Linear(num_features, 2)
        self.act = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.pretrained_model(input)
        y = self.act(x)
        return y
