import numpy as np
import os
import torch
from torch.utils.data import DataLoader

from LSIM.dataset_distance import *
from LSIM.distance_model import *
from LSIM.distance_model_non_siamese import *
from LSIM.loss import *
from LSIM.trainer import *


# SETUP FOR DATA AND MODEL
os.environ["CUDA_VISIBLE_DEVICES"]="0"
useGPU = True

trainSet = DatasetDistance("Training", dataDirs=["Data/Smoke", "Data/BurgersEq", "Data/AdvDiff", "Data/Liquid"],
                                exclude=["plume1.", "plume2.", "plume11.", "plume12.",
                                    "burgersEq1.", "burgersEq2.", "burgersEq3.", "burgersEq4.",
                                    "burgersEq5.", "burgersEq6.", "burgersEq7.", "burgersEq8.",
                                    "burgersEq9.", "burgersEq10.", "burgersEq11.", "burgersEq12.",
                                    "advDiff1.", "advDiff2.", "advDiff3.", "advDiff4.", "advDiff5.",
                                    "advDiff6.", "advDiff7.", "advDiff8.", "advDiff9.", "advDiff10.",
                                    "drop1.", "drop2.", "drop3.", "drop21.", "drop22.", "drop23."],)
valSet = DatasetDistance("Validation", dataDirs=["Data/Smoke", "Data/BurgersEq", "Data/AdvDiff", "Data/Liquid"],
                                include=["plume1.", "plume2.", "plume11.", "plume12.",
                                    "burgersEq1.", "burgersEq2.", "burgersEq3.", "burgersEq4.",
                                    "burgersEq5.", "burgersEq6.", "burgersEq7.", "burgersEq8.",
                                    "burgersEq9.", "burgersEq10.", "burgersEq11.", "burgersEq12.",
                                    "advDiff1.", "advDiff2.", "advDiff3.", "advDiff4.", "advDiff5.",
                                    "advDiff6.", "advDiff7.", "advDiff8.", "advDiff9.", "advDiff10.",
                                    "drop1.", "drop2.", "drop3.", "drop21.", "drop22.", "drop23."],)

transTrain = TransformsTrain(224, normMin=0, normMax=255)
transVal = TransformsInference(224, 0, normMin=0, normMax=255)
trainSet.setDataTransform(transTrain)
valSet.setDataTransform(transVal)

trainLoader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=4)
valLoader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=4)

model = DistanceModel(baseType="lsim", initBase="pretrained", initLin=0.1, featureDistance="L2",
                frozenLayers=[], normMode="normDist", useNormUpdate=False, isTrain=True, useGPU=useGPU)
model.printNumParams()

criterion = CorrelationLoss(weightMSE=0.3, weightCorr=0.7, weightCrossCorr=0.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0)
trainer = Trainer(model, trainLoader, optimizer, criterion, 800, False)
validator = Validator(model, valLoader, criterion)


# ACTUAL TRAINING
print('Starting Training')

if model.normMode != "normUnit":
    trainer.normCalibration(1, stopEarly=0)

for epoch in range(0, 40):
    if epoch % 5 == 1:
        validator.validationStep()

    trainer.trainingStep(epoch+1)

    model.save("Models/TrainedLSiM_tmp.pth", override=True, noPrint=True)

print('Finished Training')
model.save("Models/TrainedLSiM.pth")