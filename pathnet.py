import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class PathNet(nn.Module):

    def __init__(self, genu_class = 158, speice_classes = 2608):
        super(PathNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc7out = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.genuout =nn.Sequential(
            nn.Linear(4096, 158),
            nn.LogSoftmax(),
        )
        self.speiceout =nn.Sequential( 
            nn.Linear(4096, 2608),
            nn.LogSoftmax(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        a = self.fc7out(x)
        g = self.genuout(a)
        s = self.speiceout(a)
        return g,s


def pathnet(pretrained=False, **kwargs):
    model = PathNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model
