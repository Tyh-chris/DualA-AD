import torch
import torch.nn as nn
import torchvision

from mae.optim.projectionvgg import ProjectionNet


class VGGEncoder(nn.Module):
    """
    VGG Encoder used to extract feature representations for e.g., perceptual losses
    """
    def __init__(self, layers=[1, 6, 11, 20]):
        super(VGGEncoder, self).__init__()
        vgg = torchvision.models.vgg19(pretrained=True).features

        # vgg=self.getmodel()

        self.encoder = nn.ModuleList()
        temp_seq = nn.Sequential()
        for i in range(max(layers) + 1):
            temp_seq.add_module(str(i), vgg[i])
            if i in layers:
                self.encoder.append(temp_seq)
                temp_seq = nn.Sequential()

    def getmodel(self):
        modelname = '/home/cquml/tyh/workspace/mycode/Second/MAE_AB_3_CL/mae/model_pth/model-zhanglab-AnatPaste-vgg-best.tch'
        # print(f"loading model {modelname}")
        head_layers = [512] * 1 + [128]
        # print(head_layers)
        # print(modelname)
        # vgg = torchvision.models.vgg19(pretrained=False)
        # weights = torch.load('models/'+modelname+'.tch')
        weights = torch.load(modelname)
        # classes = weights["out.weight"].shape[0]
        model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=2)
        model.load_state_dict(weights)
        # model.to(device)
        model.eval()
        vgg = model.resnet18.features
        return vgg

    def forward(self, x):
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
        return features
