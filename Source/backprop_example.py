## IMPORTS
import numpy as np
import imageio
import torch
import torch.functional as F
from torch.utils.data import TensorDataset

from LSIM.distance_model import *
from LSIM.metrics import *


## SETUP
useGPU = True

modelLSiM = DistanceModel(baseType="lsim", isTrain=False, useGPU=useGPU)
modelLSiM.load("Models/LSiM.pth")
# freeze lsim weights
for param in modelLSiM.parameters():
    param.requires_grad = False
print()

img1 = imageio.imread("Images/plumeReference.png")[...,:3]
img2 = imageio.imread("Images/plumeA.png")[...,:3]
img3 = imageio.imread("Images/plumeB.png")[...,:3]
stacked = np.stack([img1, img2, img3])
stacked = torch.from_numpy(stacked).permute((0,3,1,2)).float()
stacked = (stacked - stacked.mean()) / stacked.std()
dataset = TensorDataset(stacked.cuda()) if useGPU else TensorDataset(stacked.cpu())
loader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)


## AUTOENCODER THAT RECONSTRUCTS FROM A COMPRESSION WITH FACTOR 4
class SimpleAE(nn.Module):
    def __init__(self):
        super(SimpleAE, self).__init__()

        eW = 64
        self.encConv = nn.Sequential(
            nn.Conv2d(3, eW, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(eW, eW, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(eW, 2*eW, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*eW, 2*eW, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*eW, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        dW = 64
        self.decConv = nn.Sequential(
            nn.Conv2d(1, dW, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2), mode='nearest'),
            nn.Conv2d(dW, dW, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2), mode='nearest'),
            nn.Conv2d(dW, 2*dW, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*dW, 2*dW, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*dW, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, data:torch.Tensor) -> torch.Tensor:
        latent = self.encConv(data)
        result = self.decConv(latent)
        return result


## LSIM LOSS FUNCTION
# input shape: [batch, channel, width, height]  -> output shape: [batch, channel]
def loss_lsim(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    # normalize inputs to [0,255]
    xMin = x.min(dim=0, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    xMax = x.max(dim=0, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    yMin = y.min(dim=0, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    yMax = y.max(dim=0, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    ref = 255 * ((x - xMin) / (xMax - xMin))
    oth = 255 * ((y - yMin) / (yMax - yMin))

    # compare each channel individually by moving them to the batch dimension
    # and adding a dummy channel dimension and lsim parameter dimension
    sizeBatch, sizeChannel = ref.shape[0], ref.shape[1]
    ref = torch.reshape(ref, (-1,1,1,ref.shape[2], ref.shape[3]))
    oth = torch.reshape(oth, (-1,1,1,oth.shape[2], oth.shape[3]))

    # clone channel dimension as lsim compares 3-channel data
    ref = ref.expand(-1,-1,3,-1,-1)
    oth = oth.expand(-1,-1,3,-1,-1)

    inDict = {"reference": ref, "other": oth}
    distance = modelLSiM(inDict)

    # move results from each original channel back into a new channel dimension
    distance = torch.reshape(distance, (sizeBatch, sizeChannel))

    return distance


## OPTIMIZATION
model = SimpleAE().cuda() if useGPU else SimpleAE().cpu()
parameters = filter(lambda p: p.requires_grad, model.parameters())
print("Parameter: %d" % sum([np.prod(p.size()) for p in parameters]))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i in range(200):
    for j, sample in enumerate(loader, 0):

        optimizer.zero_grad()
        gt = sample[0]
        rec = model(gt)
        loss = F.mse_loss(rec, gt) + loss_lsim(rec, gt).mean()

        loss.backward()
        optimizer.step()

        print("[%2d, %2d]  %1.5f" % (i,j,loss.item()))


## VISUALIZE RECONSTRUCTIONS
with torch.no_grad():
    model.eval()
    names = ["recPlumeReference", "recPlumeA", "recPlumeB"]
    for i, sample in enumerate(dataset, 0):
        gt = sample[0].unsqueeze(0)
        rec = model(gt)

        rec = rec.permute((2,3,1,0)).squeeze().cpu().numpy()
        recMin = np.min(rec)
        recMax = np.max(rec)
        rec = 255 * ((rec - recMin) / (recMax - recMin))
        imageio.imwrite("Results/%s.png" % names[i], rec.astype(np.uint8))
        print(rec.shape)
