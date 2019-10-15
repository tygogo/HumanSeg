from data import ProtraitData
from model import AAA
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim
import torch.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 50


def train(model, device, loader, opt, epochs):
    model.train()
    for epoch in range(epochs):
        for i, (data, mask) in enumerate(loader):
            data, mask = data.float(), mask.long()
            data, mask = data.to(device), mask.to(device)

            opt.zero_grad()

            out = model(data)
            loss = criterion(out, mask)
            loss.backward()

            opt.step()

            if(i+1)%10 == 0:
                print("epoch:%d  %d/%d---%.6f"%(epoch, i*len(data), len(loader.dataset), loss.item()))
                torch.save(model.state_dict(), 'C:/Users/gogo/Desktop/parameter.pkl')


model = AAA()
# 加载
#加载预训练参数
#import torchvision.models as models
#shufflenet_pretrained = models.shufflenet_v2_x1_0(pretrained=True)
#pretrained_dict = shufflenet_pretrained.state_dict()
#my_dict = model.state_dict()
#pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in my_dict}
#my_dict.update(pretrained_dict)
#model.load_state_dict(my_dict)

model.load_state_dict(torch.load('C:/Users/gogo/Desktop/parameter.pkl', map_location=DEVICE))

model = model.float().to(DEVICE)
dataset = ProtraitData(None)

loader = DataLoader(dataset,batch_size=10, shuffle=True)
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(),lr=0.0001)



# train(model, DEVICE, loader, opt, EPOCH)



