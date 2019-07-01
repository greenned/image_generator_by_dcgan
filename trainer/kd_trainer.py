import torch
from gensim.models import Word2Vec
from configs.kd_configs import *
from configs.crawl_config import *
from konlpy.tag import Okt
from dbmanager import dbManager
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader,TensorDataset
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
from statistics import mean
import time, operator
from collections import Counter
from utils import parse_reg

# 64차원으로 학습을 시키고 빈도수 기준 상위 64개의 단어를 가져온다.
# 64개가 안되면 패딩 0 

class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class FeatureMaker(dbManager):
    def __init__(self, modelpath, modelname, train_data_path):
        self.batch_size = BATCH_SIZE
        super().__init__(HOST, PORT, DB, COLLECTION_NAME)
        self.w2v_model = Word2Vec.load("{}/{}.model".format(modelpath, modelname))
        self.target = self.get_pair(KEY_NAME, VALUE_NAME)
        self.loader = self.get_loader(train_data_path)
    
    def get_loader(self, train_data_path):
        img_data = ImageFolderWithPaths("/home/ysmetal/google-drive/workspace/KANDINSKY/data/",
                        transform=transforms.Compose([
                            transforms.Resize(RESIZE_SIZE),
                            transforms.RandomCrop(CROP_SIZE),
                            transforms.ToTensor()
                        ]))
        img_loader = DataLoader(img_data, batch_size=self.batch_size, shuffle=True)
        return img_loader

    def get_embeddings(self, target_dict):
        okt = Okt()
        new_dict = dict()
        for key, value in target_dict.items():
            counter = dict(Counter(okt.nouns(value)))
            counter = sorted(counter.items(), key=operator.itemgetter(1), reverse=True)
            dim_64_list = list()
            # 배치사이즈만큼 단어 모으기
            for num, (sub_key, sub_value) in enumerate(counter):
                if num >= self.batch_size :
                    break
                else:
                    dim_64_list.append(sub_key)
            # 배치사이즈 만큼 단어가 없으면 None
            if len(dim_64_list) < self.batch_size:
                for i in range(self.batch_size - len(dim_64_list)):
                    dim_64_list.append(None)
            
            vector_list = list()
            for word in dim_64_list:
                if word is not None:
                    try:
                        vector = self.w2v_model.wv[word]
                        vector_list.append(vector)
                    except KeyError:
                        vector_list.append([0]*self.batch_size)
                else:
                    vector_list.append([0]*self.batch_size)
            new_dict[key] = vector_list
        return new_dict

class GNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(NZ, NGF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(NGF, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        out = self.main(x)
        return out

class DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF * 8, 1, 4, 1, 0, bias=False) # 14
        )
    def forward(self, x):
        out = self.main(x)
        return out.squeeze()


class DCGAN:
    def __init__(self):
        self.batch_size = BATCH_SIZE
        self.d = DNet().to("cuda:0")
        self.g = GNet().to("cuda:0")
        self.opt_d = optim.Adam(self.d.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.opt_g = optim.Adam(self.g.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.ones = torch.ones(self.batch_size).to("cuda:0")
        self.zeros = torch.zeros(self.batch_size).to("cuda:0")
        self.loss_f = nn.BCEWithLogitsLoss()
        self.fixed_z = torch.randn(self.batch_size, NZ, 1, 1).to("cuda:0")

    def train_dcgan(self, g, d, opt_g, opt_d, loader):
        log_loss_g = []
        log_loss_d = []
        for real_img, _ in tqdm(loader):
            batch_len = len(real_img)
            # 실제 이미지 GPU복사
            real_img = real_img.to("cuda:0")
            # print("real_img shape : ", real_img.shape)

            # 가짜 이미지를 난수와 생성 모델을 사용해 만듬
            z = torch.randn(batch_len, NZ, 1, 1).to("cuda:0")
            fake_img = g(z)
            # print("fake_img shaep :", fake_img.shape)
            
            # 나중을 위해 가짜 이미지값만 별도 저장
            fake_img_tensor = fake_img.detach()

            # 가짜 이미지에 대한 생성 모델의 평가 함수 계산
            out = d(fake_img)
            # print("d shape : ", out.shape)
            loss_g = self.loss_f(out, self.ones[:batch_len])
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
            loss_d_real = self.loss_f(real_out, self.ones[:batch_len])

            # 파이토치에서는 동일 텐서를 포함한 계산 그래프에 2회 backward불가
            # 저장된 텐서를 사용해서 불필요한 계산 생략
            fake_img = fake_img_tensor

            # 가짜 이미지에 대한 식별 모델 평가 함수 계산
            fake_out = d(fake_img_tensor)
            loss_d_fake = self.loss_f(fake_out, self.zeros[:batch_len])

            # 진위 평가 함수의 합계
            loss_d = loss_d_real + loss_d_fake
            log_loss_d.append(loss_d.item())

            # 식별 모델의 미분 계산과 파라미터 갱신
            d.zero_grad(), g.zero_grad()
            loss_d.backward()
            opt_d.step()

        return mean(log_loss_g), mean(log_loss_d)

def main(debug=True):
    dg = DCGAN()
    fm = FeatureMaker(MODEL_PATH, MODEL_NAME, TRAIN_DATA_PATH)

    for epoch in range(EPOCH):
        for inputs, _, paths in fm.loader:
            #parse_reg(FILE_REG, )
            print(len(inputs))
            print("-"*100)
            print("-"*100)
            print(len(paths))
            time.sleep(30)

if __name__ == "__main__":
    main()
    # fm = FeatureMaker(MODEL_PATH, MODEL_NAME)
    # new_dict = fm.get_embeddings(fm.target)

    # for key, value in new_dict.items():
    #     print(key ,": ", len(value))

    # dataset = ImageFolderWithPaths(train_data_path,
    #                     transform=transforms.Compose([
    #                         transforms.Resize(80),
    #                         transforms.RandomCrop(64),
    #                         transforms.ToTensor()
    #                     ]))
    # dataloader = DataLoader(dataset)
    # print("hi")
    # for inputs, labels, paths in dataloader:
    #     print(inputs, labels, paths)
    #     time.sleep(30)

# word = "악몽"
# model = Word2Vec.load("{}/{}.model".format(modelpath, modelname))
# a = model.most_similar("사회")
# b = model.
# print(a)

