import torch, time
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader,TensorDataset
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
from statistics import mean

batch_size = 64
ndf = 32

class ImageFolderWithPaths(ImageFolder):
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
    # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
    # the image file path
        path = self.imgs[index][0]
    # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


output_path = "/home/ysmetal/google-drive/workspace/KANDINSKY/output"
# img_data = ImageFolder("/home/ysmetal/google-drive/workspace/KANDINSKY/data/",
#                         transform=transforms.Compose([
#                             transforms.Resize(256),
#                             transforms.RandomCrop(128),
#                             transforms.ToTensor()
#                         ]))

# img_loader = myImageFolder(img_data, batch_size=batch_size, shuffle=True)



data_dir = "/home/ysmetal/google-drive/workspace/KANDINSKY/data/"
dataset = ImageFolderWithPaths(data_dir) # our custom dataset
dataloader = DataLoader(dataset)
# iterate over data
# for inputs, labels, paths in dataloader:
#     print(inputs, labels, paths)
    
output_path = "/home/ysmetal/google-drive/workspace/KANDINSKY/output"
img_data = ImageFolder("/home/ysmetal/google-drive/workspace/KANDINSKY/data/",
                        transform=transforms.Compose([
                            transforms.Resize(64),
                            transforms.RandomCrop(64),
                            transforms.ToTensor()
                        ]))

img_loader = DataLoader(img_data, batch_size=batch_size, shuffle=True)

# for num, (xx, _) in tqdm(enumerate(img_loader)):
#     if num >= 10:
#         break
#     else:
#         save_image(xx, "{}/{}.jpg".format("/home/ysmetal/google-drive/workspace/KANDINSKY/output/test", num))
# print("done")


# torch.Size([64, 3, 128, 128])
# strides * (input_size-1) + kernel_size - 2*padding

nz = 100
ngf = 32
D = nn.Sequential(
    nn.ConvTranspose2d(nz, ngf * 8, 6, 1, 1, bias=False), # torch.Size([64, 256, 4, 4]),
    nn.ConvTranspose2d(ngf * 8, ngf * 4, 6, 4, 1, bias=False), # torch.Size([64, 128, 8, 8])
    nn.ConvTranspose2d(ngf * 4, ngf * 2, 6, 4, 1, bias=False), # torch.Size([64, 64, 16, 16])
    nn.ConvTranspose2d(ngf * 2, ngf, 6, 4, 1, bias=False), # torch.Size([64, 32, 32, 32])
    nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False) # torch.Size([64, 3, 64, 64]) 
)
 # torch.Size([64, 100, 1, 1])
for xx, _ in img_loader:
    z = torch.randn(64, nz, 1, 1)
    print(z.shape)
    print("-"*100)
    print(D(z).shape) 
    time.sleep(30)




# 배치 : 64 , 크롭 : 64 torch.Size([64, 3, 64, 64])
# 배치 : 128 , 크롭 : 64 torch.Size([128, 3, 64, 64])
# 배치 : 128 , 크롭 128 torch.Size([128, 3, 128, 128])
