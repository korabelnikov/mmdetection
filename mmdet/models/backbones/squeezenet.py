import logging

from collections import OrderedDict

from mmcv.runner import load_state_dict

from ..registry import BACKBONES
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models.squeezenet import Fire, init


class _SqueezeNet(nn.Module):

    def __init__(self, version='1_0', num_classes=1000, classifier=True,
                 original_padding=False):
        """

        TODO: there is a copy from torch vision. It was impossible to change some param cause
        it is hard-coded as constant. Changing and rebuilding of vision is also not a good idea for now.

        :param version:
        :param num_classes:
        """
        super(_SqueezeNet, self).__init__()
        self.version = version
        self.num_classes = num_classes
        if original_padding:
            padding = 0
        else:
            padding = 1
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2, padding_mode='same', padding=padding),  # size/2
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # size/4
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # size/8
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # size/16
                Fire(512, 64, 256, 256),
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding_mode='same', padding=padding),  # size/2
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # size/4
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # size/8
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # size/16
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        if classifier:
            # Final convolution is initialized differently from the rest
            self._final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)

            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                self._final_conv,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        else:
            self._final_conv = None
            self.classifier = None

    def init_weights(self, pretrained=None):
        if pretrained:
            model_urls = {
                'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
                'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
            }
            arch = 'squeezenet' + self.version
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                  progress=True)
            logger = logging.getLogger()
            load_state_dict(self, state_dict, strict=False, logger=logger)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m is self._final_conv:
                        init.normal_(m.weight, mean=0.0, std=0.01)
                    else:
                        init.kaiming_uniform_(m.weight)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        if self.classifier is not None:
            x = self.classifier(x)
            return torch.flatten(x, 1)
        return x


@BACKBONES.register_module
class SqueezeNet(_SqueezeNet):
    """SqueezeNet backbone.

    Args:
        #out_indices (Sequence[int]): Output from which stages.
        frozen_layers (int): Stages to be frozen (stop grad and set eval mode).
            0 means not freezing any parameters.
    """
    def __init__(self, version='1_1', frozen_layers=0, original_padding=False):

        super(SqueezeNet, self).__init__(version, 0, False, original_padding)
        for l in self.features[:frozen_layers]:
            l.eval()
            for param in l.parameters():
                param.requires_grad = False
            print('{} first layers have been frozen'.format(frozen_layers))

    def forward(self, x):
        outs = []
        assert self.version == '1_1'
        for i, l in enumerate(self.features):
            x = l(x)
            if i in [4, 7, 12]:
                outs.append(x)
        return outs


if __name__ == '__main__':
    # check original model
    model = _SqueezeNet('1_1')
    model.init_weights(True)

    # Download an example image from the pytorch website
    import urllib

    url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)

    from PIL import Image
    from torchvision import transforms

    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    print(torch.nn.functional.softmax(output[0], dim=0))
    import numpy as np

    assert np.argmax(output[0]) == 258  # 258: 'Samoyed, Samoyede',
