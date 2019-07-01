import torch
from torch import nn
from torchvision.utils import save_image

nz = 100
ngf = 32

class GNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.main(x)
        return out

modelpath = "/home/ysmetal/google-drive/workspace/KANDINSKY/output"
modelname = "g_080.prm"
outputpath = "/home/ysmetal/google-drive/workspace/KANDINSKY/output/test"
PATH = "{}/{}".format(modelpath, modelname)


model = GNet() #*args, **kwargs
model.load_state_dict(torch.load(PATH))
model.to("cuda:0")
model.eval()

z = torch.randn(64, nz, 1, 1).to("cuda:0")

gen = model(z)
save_image(gen[0], "{}/{}.jpg".format(outputpath,"test"))
print(gen)