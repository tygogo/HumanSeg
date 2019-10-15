import wx
import cv2
import threading, time,random
import torch
import numpy as np
from model import AAA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AAA()
model.load_state_dict(torch.load('./parameter.pkl', map_location=DEVICE))

def getRst(model, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    img2 = img.astype("float") / 255.0
    img2 = img2.transpose(2,0,1)
    img2 = np.expand_dims(img2, axis=0)
    with torch.no_grad():
        input_data = torch.from_numpy(img2).float()

        result = model(input_data)[0]
        mask = result.max(0)[1]
        mask = mask.detach().numpy().astype(np.uint8)
        mask_img = cv2.merge([mask,mask,mask]) * 255

    return img, mask_img


class Canvas(wx.Panel):
    def __init__(self, parent):
        super(Canvas, self).__init__(parent)
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE,self.on_size)

        self.bitmap1 = wx.Bitmap()
        self.bitmap2 = wx.Bitmap()
        
    def on_size(self,event):
        self.Refresh()

    def on_paint(self,event):
        # w,h = self.GetClientSize()
        dc = wx.AutoBufferedPaintDC(self)
        dc.Clear()
        dc.DrawBitmap(self.bitmap1,0,0)
        dc.DrawBitmap(self.bitmap2,224,0)
        
        # dc.DrawLine(0,0,random.randint(0,w),random.randint(0,h))
        # dc.SetPen(wx.Pen(wx.RED, 5))
    

app = wx.App()
wd = wx.Frame(None,size=(480,360))
canvas = Canvas(wd)

def videoCapture():
    capture = cv2.VideoCapture(0)
    while True:
        _, img = capture.read()    #720  1280  变成320 240
        h,w,c = img.shape
        scale = 320.0/h
        img = cv2.resize(img,(0,0),fx=scale, fy=scale)
        

        targe_h, targe_w, _ = img.shape
        img = img[:, (targe_w - 240)//2 : (targe_w - 240)//2 +240,:]
        print(img.shape)

        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.show()
        # sleep(10)
        img = img[:,::-1,:]
        img, rst = getRst(model, img)
        img = cv2.resize(img,(240,320))
        rst = cv2.resize(rst,(240,320))

        x = np.array(img)
        x[rst == 0] = 0
        
        canvas.bitmap1 = wx.Bitmap.FromBuffer(240,320,img)
        canvas.bitmap2 = wx.Bitmap.FromBuffer(240,320,x)

        
        wx.CallAfter(canvas.Refresh)

t = threading.Thread(target=videoCapture)
t.setDaemon(True)
t.start()
# canvas.Bind(wx.EVT_CLOSE,lambda e: t)

wd.Show()
app.MainLoop()