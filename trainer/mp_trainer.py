import torch, time
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader,TensorDataset
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
from statistics import mean

batch_size = 32
nz = 100
ngf = 32
ndf = 32
img_size = 512

output_path = "/home/ysmetal/google-drive/workspace/KANDINSKY/output/all_512_pixel"
img_data = ImageFolder("/home/ysmetal/google-drive/workspace/KANDINSKY/data/",
                        transform=transforms.Compose([
                            transforms.Resize(img_size),
                            transforms.CenterCrop(img_size),
                            transforms.ToTensor()
                        ]))

img_loader = DataLoader(img_data, batch_size=batch_size, shuffle=True)

class GNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 6, 4, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 6, 4, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, 6, 4, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.main(x)
        return out

class DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 6, 4, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 6, 4, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 6, 4, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 6, 4, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 1, bias=False) # 14
        )
    def forward(self, x):
        out = self.main(x)
        return out.squeeze()

d = DNet().to("cuda:0")
g = GNet().to("cuda:0")

opt_d = optim.Adam(d.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_g = optim.Adam(g.parameters(), lr=0.0002, betas=(0.5, 0.999))

ones = torch.ones(batch_size).to("cuda:0")
zeros = torch.zeros(batch_size).to("cuda:0")
loss_f = nn.BCEWithLogitsLoss()

fixed_z = torch.randn(batch_size, nz, 1, 1).to("cuda:0")

def train_dcgan(g, d, opt_g, opt_d, loader):
    log_loss_g = []
    log_loss_d = []
    for real_img, _ in tqdm(loader):
        batch_len = len(real_img)
        # 실제 이미지 GPU복사
        real_img = real_img.to("cuda:0")
        # print("real_img shape : ", real_img.shape)

        # 가짜 이미지를 난수와 생성 모델을 사용해 만듬
        z = torch.randn(batch_len, nz, 1, 1).to("cuda:0")
        fake_img = g(z)
        #print("fake_img shape :", fake_img.shape)
        
        # 나중을 위해 가짜 이미지값만 별도 저장
        fake_img_tensor = fake_img.detach()

        # 가짜 이미지에 대한 생성 모델의 평가 함수 계산
        out = d(fake_img)
        # print("d shape : ", out.shape)
        loss_g = loss_f(out, ones[:batch_len])
        log_loss_g.append(loss_g.item())

        # 계산 그래프가 생성 모델과 식별 모델 양쪽에 의존하므로 양쪽모두 경사하강 끝낸 후 
        # 미분계산과 파라미터 갱신을 실시
        d.zero_grad(), g.zero_grad()
        loss_g.backward()
        opt_g.step()

        # 실제 이미지에 대한 식별 모델 평가 함수 계산
        real_out = d(real_img)
        # print("real_out shape :" , real_out.shape)
        # print("-"*100)
        # print(ones[:batch_len])
        loss_d_real = loss_f(real_out, ones[:batch_len])

        # 파이토치에서는 동일 텐서를 포함한 계산 그래프에 2회 backward불가
        # 저장된 텐서를 사용해서 불필요한 계산 생략
        fake_img = fake_img_tensor

        # 가짜 이미지에 대한 식별 모델 평가 함수 계산
        fake_out = d(fake_img_tensor)
        loss_d_fake = loss_f(fake_out, zeros[:batch_len])

        # 진위 평가 함수의 합계
        loss_d = loss_d_real + loss_d_fake
        log_loss_d.append(loss_d.item())

        # 식별 모델의 미분 계산과 파라미터 갱신
        d.zero_grad(), g.zero_grad()
        loss_d.backward()
        opt_d.step()

    return mean(log_loss_g), mean(log_loss_d)

# 모델 로드
# g_model_path = "/home/ysmetal/google-drive/workspace/KANDINSKY/output/g_020.prm"
# d_model_path = "/home/ysmetal/google-drive/workspace/KANDINSKY/output/d_020.prm"

# g = GNet().to("cuda:0")
# g = g.load_state_dict(torch.load(g_model_path))
# print(g)
# d = DNet().to("cuda:0")
# d = d.load_state_dict(torch.load(d_model_path))
# opt_d = optim.Adam(d.parameters(), lr=0.0002, betas=(0.5, 0.999))
# opt_g = optim.Adam(g.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in tqdm(range(300)):
    train_dcgan(g, d, opt_g, opt_d, img_loader)
    # 10회마다 결과 저장

    if epoch % 5 == 0:
        torch.save(
            g.state_dict(),
            "{}/g_{:03d}.prm".format(output_path, epoch),
            pickle_protocol=4
        )

        torch.save(
            d.state_dict(),
            "{}/d_{:03d}.prm".format(output_path, epoch),
            pickle_protocol=4
        )

        generated_img = g(fixed_z)
        save_image(generated_img,
                "{}/{:03d}.jpg".format(output_path,epoch))