import torch
import torch.nn as nn
import torch.nn.functional as F

class ShuffleResBlock(nn.Module):
    def __init__(self, in_c:int, out_c:int, groups:int, stride = 1):
        super(ShuffleResBlock,self).__init__()
        self.stride = stride
        if in_c == 24:
            groups = 1
        self.groups = groups

        mid_channel = int(in_c / 4)
        self.mid_channel = mid_channel

        # print(in_c)
        self.conv1 = nn.Conv2d(in_c, mid_channel, kernel_size=1, stride=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.DWConv = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=stride, groups=mid_channel, bias=False,padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channel)
        self.conv2 = nn.Conv2d(mid_channel, out_c, kernel_size=1, stride=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)

        self.pool = nn.AvgPool2d(3, 2, padding=1)
        self.relu = nn.ReLU(inplace=True)


    def channelShuffle(self, data_in:torch.Tensor, groups:int):
        batch_size, channel_size, h, w = data_in.shape
        return data_in.view(batch_size, groups, int(channel_size/groups), h,w).permute(0,2,1,3,4).contiguous().view(batch_size, channel_size, h, w)


    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.channelShuffle(out, self.groups)
        out = self.bn1(out)
        out = self.DWConv(out)
        out = self.bn2(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.stride == 2:
            shortcut = self.pool(shortcut)
            out =  torch.cat([out, shortcut],dim=1)
        else:
            out = out + shortcut

        return self.relu(out)


class ShuffleFCN(nn.Module):
    def __init__(self):
        super(ShuffleFCN,self).__init__()
        self.conv1 = nn.Conv2d(3,24,kernel_size=3,stride=2,padding=1)  # 2s
        self.maxpooling = nn.MaxPool2d(3,2,padding=1) # 4s

        self.stage2 = self._makeStage(24,240,3) # 8s
        self.stage3 = self._makeStage(240,480,7) # 16s
        self.stage4 = self._makeStage(480,960,3) # 32s

        self.deconv32s = nn.ConvTranspose2d(960,480,kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn32s = nn.BatchNorm2d(480)
        self.deconv16s = nn.ConvTranspose2d(480,240,kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn16s = nn.BatchNorm2d(240)
        self.deconv8s = nn.ConvTranspose2d(240,24,kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn8s = nn.BatchNorm2d(24)
        self.deconv4s = nn.ConvTranspose2d(24,24,kernel_size=3, stride=4, padding=1, output_padding=3)
        self.bn4s = nn.BatchNorm2d(24)
        # self.deconv4s = nn.ConvTranspose2d(24,24,kernel_size=3, stride=4, padding=1)
        # self.bn4s = nn.BatchNorm2d(24)
        self.cls = nn.Conv2d(24,2,kernel_size=1,stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.maxpooling(self.conv1(x))  #4s
        x8s = self.stage2(x)
        x16s = self.stage3(x8s)
        x32s = self.stage4(x16s)

        upto16 = self.bn32s(self.relu(self.deconv32s(x32s)))
        x16s = x16s + upto16
        upto8 = self.bn16s(self.relu(self.deconv16s(x16s)))
        x8s = x8s + upto8
        upto4 = self.bn8s(self.relu(self.deconv8s(x8s)))

        x = self.bn4s(self.relu(self.deconv4s(upto4)))
        x = self.cls(x)
        return x

    def _makeStage(self, in_c:int, out_c:int,block_num:int):
        s = nn.Sequential()
        for i in range(block_num):
            if i == 0:
                block = ShuffleResBlock(in_c, out_c-in_c, groups=3, stride=2)
            else:
                block = ShuffleResBlock(out_c, out_c, groups=3, stride=1)
            s.add_module("ShuffleBlock #" + str(i+1), block)
        return s

        # #14
        # self.up4conv1 = nn.Conv2d(464,232,kernel_size=1,stride=1)
        # self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        # #add
        # self.up4conv3 = nn.Conv2d(232,116,kernel_size=3,stride=1,padding=1)

        # #28
        # self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        # #add
        # self.up3conv3 = nn.Conv2d(116,24,kernel_size=3,stride=1,padding=1)
        
        # #56
        # self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        # #add
        # self.up2conv3 = nn.Conv2d(24,24,kernel_size=3,stride=1,padding=1)

        # #224
        # self.up1 = nn.UpsamplingBilinear2d(scale_factor=4)
        # #add 
        # self.up2conv3 = nn.Conv2d(24,24,kernel_size=3,stride=1,padding=1)

class AAA(nn.Module):
    def __init__(self):
        super(AAA, self).__init__()
        import torchvision.models as models
        model = models.shufflenet_v2_x1_0()
        self.conv1 = model.conv1  #112x112   24                               2s
        self.maxpool = model.maxpool  #56x56   24                             4s
        self.stage2 = model.stage2  #28x28    58 残差连接,,两个58 116          8s
        self.stage3 = model.stage3  #14x14    232                            16s
        self.stage4 = model.stage4  #7x7      464                            32s

        self.deconv32s = nn.ConvTranspose2d(464,232,kernel_size=3, stride=2, padding=1, output_padding=1)   #变16s
        self.bn32s = nn.BatchNorm2d(232)

        self.deconv16s = nn.ConvTranspose2d(232,116,kernel_size=3, stride=2, padding=1, output_padding=1)   #变8s
        self.bn16s = nn.BatchNorm2d(116)

        self.deconv8s = nn.ConvTranspose2d(116,24,kernel_size=3, stride=2, padding=1, output_padding=1)     #变成4s
        self.bn8s = nn.BatchNorm2d(24)

        self.deconv4s = nn.ConvTranspose2d(24,24,kernel_size=3, stride=2, padding=1, output_padding=1)      #变2s
        self.bn4s = nn.BatchNorm2d(24)

        self.deconv2s = nn.ConvTranspose2d(24,64,kernel_size=3,stride=2, padding=1,output_padding=1)        #变1s
        self.bn2s = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.cls = nn.Conv2d(64,2,kernel_size=1,stride=1)

    def forward(self, x):
        x2 = self.conv1(x)   #2s
        x4 = self.maxpool(x2) #4s
        x8 = self.stage2(x4)  #8s
        x16 = self.stage3(x8)  #16s
        x32 = self.stage4(x16)  #32s    7

        up16 = self.relu(self.bn32s(self.deconv32s(x32)))   #  14
        up16 = up16 + x16

        up8 = self.relu(self.bn16s(self.deconv16s(up16)))  #  28
        up8 = up8 + x8
        
        up4 = self.relu(self.bn8s(self.deconv8s(up8)))  #  56
        up4 = up4 + x4

        up2 = self.relu(self.bn4s(self.deconv4s(up4)))  #  112
        # print(up2.shape)
        up2 = up2 + x2

        x = self.relu(self.bn2s(self.deconv2s(up2)))  # 224

        x = self.cls(x)
        return x



if __name__ == "__main__":
    x = torch.randn(1,3,224,224)
    model = AAA()
    print(model)
    y = model(x)
    print(y.shape)


    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))




#加载
#加载预训练参数
    # import torchvision.models as models
    # shufflenet_pretrained = models.shufflenet_v2_x1_0(pretrained=True)
    # pretrained_dict = shufflenet_pretrained.state_dict()
    # my_dict = model.state_dict()
    # pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in my_dict}
    # my_dict.update(pretrained_dict)
    # model.load_state_dict(my_dict)

    # torch.save(model.state_dict(), 'C:/Users/ayogg/Desktop/parameter.pkl')




# G=8    5次下采样


# class ShuffleUNet(nn.Module):
#     def __init__(self, block_config:list=[(4,384),(8,768),(4,1536)], groups = 8):
#         super(ShuffleUNet, self).__init__()

#         self.conv1 = nn.Conv2d(3, 24, kernel_size=7, stride=2, padding=3, bias=False)
#         self.maxpooling = nn.MaxPool2d(kernel_size=3,stride=2, padding=1)
#         self.cur_channel = 24

#         channel_book = [24,24]

#         # down
#         self.downblocks = []
#         stage_count=1
#         for block_num, channel_num in block_config:
#             stage_sequence = nn.Sequential()
#             for i in range(block_num):
#                 block = None
#                 if i == 0:
#                     # print(self.cur_channel)
#                     block = ShuffleResBlock(in_c=self.cur_channel, out_c=channel_num - self.cur_channel, groups=groups, stride=2)
#                     self.cur_channel = channel_num
#                 else:
#                     block = ShuffleResBlock(self.cur_channel, self.cur_channel, groups=groups, stride=1)
#                 stage_sequence.add_module("##" + str(i),block)
#             self.add_module("stage #" + str(stage_count) , stage_sequence)
#             self.downblocks.append(stage_sequence)
#             stage_count = stage_count + 1
#             channel_book.append(self.cur_channel)
        
#         channel_book.pop()

#         # up
#         self.upblocks = []
#         stage_count=1
#         for up_to_channel in reversed(channel_book):
#             sequence = nn.Sequential()
#             sequence.add_module("upsampling #" + str(stage_count), nn.UpsamplingBilinear2d(scale_factor=2))
#             sequence.add_module("conv3 #" + str(stage_count), nn.Conv2d(self.cur_channel, up_to_channel, 3, 1, padding=1))
#             self.cur_channel = up_to_channel
#             sequence.add_module('norm #' + str(stage_count), nn.BatchNorm2d(up_to_channel))
#             sequence.add_module('relu', nn.ReLU(inplace=True))
#             self.add_module("upsample stage #" + str(stage_count), sequence)
#             self.upblocks.append(sequence)
#             stage_count = stage_count + 1

#     def forward(self, x):
#         x1 = self.conv1(x)
#         x = self.maxpooling(x1)
#         upto = [x1,x]
#         for block in self.downblocks:
#             x = block(x)
#             upto.append(x)

#         upto.pop()
#         for block in self.upblocks:
#             x = block(x)
#             a = upto.pop()
#             x = x + a
        
#         x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
#         x = nn.Conv2d(24,1,1,1)(x)
#         return x