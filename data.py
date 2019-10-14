from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
import cv2
import random


class ProtraitData(Dataset):
    def __init__(self):
        super(ProtraitData, self).__init__()
        # self.data_dir = "D:/Portrait/src/"
        # self.mask_dir = "D:/Portrait/alpha/"
        self.data_dir = "D:/data/HUMAN/src/"
        self.mask_dir = "D:/data/HUMAN/alpha/"
        # self.size = len(os.listdir(self.data_dir))
        self.fileList = os.listdir(self.data_dir)

    def __getitem__(self, idx):
        filename = self.fileList[idx]
        data_path = os.path.join(self.data_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        data = cv2.imread(data_path)
        mask = cv2.imread(mask_path)

        data, mask = self.transform(data,mask)

        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        mask[mask > 0] = 1

        data = data.transpose(2, 0, 1)

        data, mask = torch.from_numpy(data), torch.from_numpy(mask)
        return data, mask

    def __len__(self):
        return len(self.fileList)

    # 原始是600X800
    def transform(self, img, mask):
        # resize to 250
        # img = cv2.resize(img, (300, 300))
        # mask = cv2.resize(mask, (300, 300))
        
        # short edge to 300
        h, w, _ = img.shape
        scale = 255.0 / min(w, h)
        target_w = int(w * scale)
        target_h = int(h * scale)
    
        img = cv2.resize(img, (target_w, target_h))
        mask = cv2.resize(mask, (target_w, target_h))

        target_h, target_w, _ = img.shape
        # random crop to 224
        start_y = random.randint(0, target_h-224)
        start_x = random.randint(0, target_w-224)
        img = img[start_y: start_y+224, start_x:start_x+224]
        mask = mask[start_y: start_y+224, start_x:start_x+224]

        # random flip
        if random.random() > 0.5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        return img, mask


class ProtraitValidataData(Dataset):
    def __init__(self):
        super(ProtraitData, self).__init__()
        self.data_dir = "D:/Vali"
        self.mask_dir = "D:/Portrait/alpha/"
        # self.data_dir = "C:/Users/ayogg/Desktop/Portrait/src/"
        # self.mask_dir = "C:/Users/ayogg/Desktop/Portrait/alpha/"

        self.size = len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        idx = idx + 1
        name = "{:0>5d}".format(idx)

        data_path = os.path.join(self.data_dir, name + ".png")
        mask_path = os.path.join(self.mask_dir, name + "_matte.png")

        data = io.imread(data_path)
        mask = io.imread(mask_path)

        data = skitransform.resize(data, (224, 224))
        mask = skitransform.resize(mask, (224, 224))

        data = data.transpose(2, 0, 1)

        mask[mask > 0] = 1

        if(self.transform != None):
            data = self.transform(data)

        data, mask = torch.from_numpy(data), torch.from_numpy(mask)
        return data, mask

    def __len__(self):
        return self.size


if __name__ == "__main__":
    dl = DataLoader(ProtraitData(), batch_size=1)
    idata = iter(dl)
    import matplotlib.pyplot as plt
    img, mask = next(idata)
    print(img.shape)
    print(mask.shape)
    print(img[0][0][0][0].dtype)
    print(mask[0][0][0].dtype)
    
    plt.subplot(1,2,1)
    img = img[0].numpy().transpose(1,2,0)
    plt.imshow(img)
    plt.subplot(1,2,2)
    mask = mask[0]
    plt.imshow(mask)
    plt.show()


