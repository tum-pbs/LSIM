## IMPORTS
import numpy as np
import imageio

# LPIPS
from PerceptualSimilarity.util import util
from PerceptualSimilarity.models import dist_model as dm

# LSiM
from LSIM.distance_model import *
from LSIM.metrics import *


## MODEL INITIALIZATION
use_gpu = True

modelLSiM = DistanceModel(baseType="lsim", isTrain=False, useGPU=use_gpu)
modelLSiM.load("Models/LSiM.pth")

modelL2 = Metric("L2")

modelSSIM = Metric("SSIM")

modelLPIPS = dm.DistModel()
modelLPIPS.initialize(model='net-lin', net='alex', use_gpu=use_gpu, spatial=False)
print()


## DISTANCE COMPUTATION
ref = imageio.imread("Images/plumeReference.png")[...,:3]
plumeA = imageio.imread("Images/plumeA.png")[...,:3]
plumeB = imageio.imread("Images/plumeB.png")[...,:3]

distA_LSiM = modelLSiM.computeDistance(ref, plumeA, interpolateTo=224, interpolateOrder=0)
distB_LSiM = modelLSiM.computeDistance(ref, plumeB, interpolateTo=224, interpolateOrder=0)

distA_L2 = modelL2.computeDistance(ref, plumeA)
distB_L2 = modelL2.computeDistance(ref, plumeB)

distA_SSIM = modelSSIM.computeDistance(ref, plumeA)
distB_SSIM = modelSSIM.computeDistance(ref, plumeB)


# convert numpy arrays to tensor for the LPIPS model
tensRef = util.im2tensor(ref)
tensPlumeA = util.im2tensor(plumeA)
tensPlumeB = util.im2tensor(plumeB)

distA_LPIPS = modelLPIPS(tensRef, tensPlumeA)
distB_LPIPS = modelLPIPS(tensRef, tensPlumeB)

# print distance results
print("LSiM   --  PlumeA: %0.4f  PlumeB: %0.4f" % (distA_LSiM, distB_LSiM))
print("L2     --  PlumeA: %0.4f  PlumeB: %0.4f" % (distA_L2, distB_L2))
print("SSIM   --  PlumeA: %0.4f  PlumeB: %0.4f" % (distA_SSIM, distB_SSIM))
print("LPIPS  --  PlumeA: %0.4f  PlumeB: %0.4f" % (distA_LPIPS, distB_LPIPS))

# distance results should look like this (on CPU and GPU):
#LSiM   --  PlumeA: 0.3791  PlumeB: 0.4433
#L2     --  PlumeA: 0.0708  PlumeB: 0.0651
#SSIM   --  PlumeA: 0.3927  PlumeB: 0.4020
#LPIPS  --  PlumeA: 0.3118  PlumeB: 0.3527
