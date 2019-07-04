from deepclustering.utils import _warnings
from torch import nn
from torchvision.models import resnet18


class Identical(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


class ResNetHead(nn.Module):
    def __init__(self, num_sub_heads, output_k):
        super().__init__()
        self.num_sub_heads = num_sub_heads
        self.heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(512, output_k), nn.Softmax(dim=1))
                for _ in range(self.num_sub_heads)
            ]
        )

    def forward(self, x):
        results = []
        for i in range(self.num_sub_heads):
            results.append(self.heads[i](x))
        return results


class ResNet50(nn.Module):
    def __init__(
        self,
        num_channel=3,
        num_sub_heads=5,
        output_k_A=70,
        output_k_B=10,
        *args,
        **kwargs
    ):
        _warnings(args, kwargs)
        super().__init__()
        self.input_convert = nn.Conv2d(num_channel, 3, kernel_size=3, padding=1)
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = Identical()
        self.head_A = ResNetHead(num_sub_heads, output_k_A)
        self.head_B = ResNetHead(num_sub_heads, output_k_B)

    def forward(self, input, head="B"):
        if head == "A":
            return self.head_A(self.resnet(self.input_convert(input)))
        else:
            return self.head_B(self.resnet(self.input_convert(input)))


if __name__ == "__main__":
    resnet = resnet50(pretrained=True)
    print()
