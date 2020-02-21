## IMPORTS
import numpy as np
import torch
import sys
import cv2

# LPIPS
from perceptual_similarity.util import util
from perceptual_similarity.models import dist_model as dm

# LSiM
from lsim.distance_model import *


## MODEL INITIALIZATION
use_gpu = True

modelLSiM = DistanceModel(baseType="lsim", dataMode="all", isTrain=False, useGPU=use_gpu)
modelLSiM.load("models/LSiM.pth")

modelL2 = dm.DistModel()
modelL2.initialize(model='l2', colorspace='Lab', use_gpu=use_gpu)

modelLPIPS = dm.DistModel()
modelLPIPS.initialize(model='net-lin',net='alex', use_gpu=use_gpu, spatial=False)
print()


## DISTANCE COMPUTATION
ref = cv2.imread("images/plumeReference.png")
plumeA = cv2.imread("images/plumeA.png")
plumeB = cv2.imread("images/plumeB.png")

distA_LSiM = modelLSiM.computeDistance(ref, plumeA)
distB_LSiM = modelLSiM.computeDistance(ref, plumeB)

# convert numpy arrays to tensor for LPIPS models
tensRef = util.im2tensor(ref)
tensPlumeA = util.im2tensor(plumeA)
tensPlumeB = util.im2tensor(plumeB)

distA_L2 = modelL2.forward(tensRef, tensPlumeA)
distB_L2 = modelL2.forward(tensRef, tensPlumeB)
distA_LPIPS = modelLPIPS.forward(tensRef, tensPlumeA)
distB_LPIPS = modelLPIPS.forward(tensRef, tensPlumeB)

# print distance results
print("LSiM   --  PlumeA: %0.4f  PlumeB: %0.4f" % (distA_LSiM, distB_LSiM))
print("L2     --  PlumeA: %0.4f  PlumeB: %0.4f" % (distA_L2, distB_L2))
print("LPIPS  --  PlumeA: %0.4f  PlumeB: %0.4f" % (distA_LPIPS, distB_LPIPS))

# distance results should look like this (on CPU and GPU):
#LSiM   --  PlumeA: 0.1405  PlumeB: 0.2484
#L2     --  PlumeA: 0.0124  PlumeB: 0.0115
#LPIPS  --  PlumeA: 0.3118  PlumeB: 0.3527
