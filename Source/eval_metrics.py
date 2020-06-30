import os
import numpy as np
import torch
import scipy.stats.stats as sciStats
import csv, datetime

# LPIPS
from PerceptualSimilarity.models import dist_model as lpipsModel

from LSIM.distance_model import *
from LSIM.distance_model_non_siamese import *
from LSIM.metrics import *


os.environ["CUDA_VISIBLE_DEVICES"]="0"

useGPU = True
batch = 8
workerThreads = 4
cutOffIndex = 10  #only compare first image to all others (instead of every pair) for better performance
#loadFile= "Results/distances.npy"
loadFile = ""
saveDir = "Results/"
mode = "spearman"
#mode = "pearson"
#mode = "pearsonMean"


# MODEL AND DATA INITIALIZATION
model1 = lpipsModel.DistModel()
model1.initialize(model='net-lin',net='alex',use_gpu=useGPU,spatial=False)

model2 = DistanceModel(baseType="alex", featureDistance="L2", frozenLayers=[0,1,2,3,4], normMode="normUnit", isTrain=False, useGPU=useGPU)
model2.load("Models/Experimental/Alex_InitRandom.pth")
model3 = DistanceModel(baseType="alex", featureDistance="L2", frozenLayers=[0,1,2,3,4], normMode="normUnit", isTrain=False, useGPU=useGPU)
model3.load("Models/Experimental/Alex_Frozen.pth")
model4 = DistanceModelNonSiamese(initBase="none", isTrain=False, useGPU=useGPU)
model4.load("Models/Experimental/NonSiamese.pth")
model5 = DistanceModel(baseType="lsimSkip", featureDistance="L2", frozenLayers=[], normMode="normDist", isTrain=False, useGPU=useGPU)
model5.load("Models/Experimental/SkipConnections.pth")

model6 = DistanceModel(baseType="lsim", featureDistance="L2", frozenLayers=[], normMode="normDist", isTrain=False, useGPU=useGPU)
model6.load("Models/Experimental/Lsim_DataNoiseless.pth")
model7 = DistanceModel(baseType="lsim", featureDistance="L2", frozenLayers=[], normMode="normDist", isTrain=False, useGPU=useGPU)
model7.load("Models/Experimental/Lsim_DataStrongNoise.pth")

model8 = DistanceModel(baseType="lsim", featureDistance="L2", frozenLayers=[], normMode="normDist", isTrain=False, useGPU=useGPU)
model8.load("Models/LSiM.pth")
print("")

metrics = [Metric("L2"), Metric("SSIM"), model1, model2, model3, model4, model5, model6, model7, model8]
names = ["L2", "SSIM", "LPIPS", "Alex_random", "Alex_frozen", "Non-Siamese", "Skip_scratch", "LSiM_noiseless", "LSiM_strongNoise", "LSiM(ours)"]

assert(len(metrics) == len(names)), "Not all models are named!"

dataSets = [DatasetDistance("Smo",  dataDirs=["Data/Smoke"],         include=["plume1.", "plume2.", "plume11.", "plume12."] ),
            DatasetDistance("Liq",  dataDirs=["Data/Liquid"],        include=["drop1.", "drop2.", "drop3.", "drop21.", "drop22.", "drop23."] ),
            DatasetDistance("Adv",  dataDirs=["Data/AdvDiff"],       include=["advDiff1.", "advDiff2.", "advDiff3.", "advDiff4.", "advDiff5.", "advDiff6.", "advDiff7.", "advDiff8.", "advDiff9.", "advDiff10."] ),
            DatasetDistance("Bur",  dataDirs=["Data/BurgersEq"],     include=["burgersEq1.", "burgersEq2.", "burgersEq3.", "burgersEq4.", "burgersEq5.", "burgersEq6.", "burgersEq7.", "burgersEq8.", "burgersEq9.", "burgersEq10.", "burgersEq11.", "burgersEq12."] ),
            DatasetDistance("AdvD", dataDirs=["Data/AdvDiffDensity"],include=["advDiff1.", "advDiff2.", "advDiff3.", "advDiff4.", "advDiff5.", "advDiff6.", "advDiff7.", "advDiff8.", "advDiff9.", "advDiff10."] ),
            DatasetDistance("LiqN", dataDirs=["Data/LiquidNoise"] ),
            DatasetDistance("Sha",  dataDirs=["Data/Shapes"] ),
            DatasetDistance("Vid",  dataDirs=["Data/Video"],         include=["scene1"] ),
            DatasetDistance("TID",  dataDirs=["Data/TID2013"],       include=["tid0"] ),
        ]
allCorIncludes = ["AdvD", "LiqN", "Sha", "Vid"]
splits = [
    #[".density", ".pressure", ".vel"],
    #[".flags", ".phi", ".vel"],
]
print()

dataLoaders = []
for dataSet in dataSets:
    dataSet.setDataTransform(TransformsInference(outputSize=224, order=0, normMin=0, normMax=255))
    dataLoaders += [DataLoader(dataSet, batch_size=batch, num_workers=workerThreads)]

# DISTANCE EVALUATION
predictions = []
targets = []
paths = []
if loadFile:
    print("Loading distances from " + loadFile)
    loaded = np.load(loadFile).item()
    assert(loaded["metrics"] == names), "Metric loading mismatch"
    loadDatasets = []
    for d in range(len(dataSets)):
        loadDatasets += [dataSets[d].name]
    assert(loaded["datasets"] == loadDatasets), "Dataset loading mismatch!"

    predictions = loaded["predictions"]
    targets = loaded["targets"]
    paths = loaded["paths"]
    print(paths)
else:
    print("Computing distances for %d metrics" % len(metrics))
    for d in range(len(dataSets)):
        dist = np.zeros([len(metrics), len(dataSets[d]), cutOffIndex])
        targ = np.zeros([1, len(dataSets[d]), cutOffIndex])
        dataPaths = []

        for s, sample in enumerate(dataLoaders[d], 0):
            print(sample["path"])#, end="\r")
            dataPaths += sample["path"]
            targ[0,s*batch:(s+1)*batch] = sample["distance"][:,0:cutOffIndex]

            ref = sample["reference"].cuda() if useGPU else sample["reference"]
            current = sample["other"].cuda() if useGPU else sample["other"]

            sampleMetrics = {"reference": ref[:,0:cutOffIndex], "other": current[:,0:cutOffIndex],
                        "distance": sample["distance"], "path": sample["path"]}

            sampleTrained = {"reference": ref, "other": current,
                        "distance": sample["distance"], "path": sample["path"]}

            with torch.no_grad():
                for m in range(len(metrics)):
                    if type(metrics[m]) is lpipsModel.DistModel:
                        for i in range(ref.shape[0]):
                            for j in range(cutOffIndex):
                                temp1 = torch.unsqueeze(ref[i,j], 0)
                                temp2 = torch.unsqueeze(current[i,j], 0)
                                dist[m,s*batch+i,j] = metrics[m].forward(temp1, temp2)

                    elif type(metrics[m]) is Metric:
                        result = metrics[m](sampleMetrics)
                        dist[m,s*batch:(s+1)*batch] = result.numpy()

                    elif type(metrics[m]) is DistanceModel or type(metrics[m]) is DistanceModelNonSiamese:
                        result = metrics[m](sampleTrained)
                        dist[m,s*batch:(s+1)*batch] = result[:,0:cutOffIndex].cpu().numpy()

        # normalize distances to comparable [0.1, 1.0] range
        dMax = np.max(dist, axis=2, keepdims=True)
        dMin = np.min(dist, axis=2, keepdims=True)
        dist = 0.9 * ((dist - dMin) / (dMax - dMin)) + 0.1

        predictions += [ np.reshape(dist, (len(metrics), -1)) ]
        targets += [ np.reshape(targ, (1, -1)) ]
        paths += [ np.repeat(np.array(dataPaths), cutOffIndex, axis=0) ]
        print("")

    # SAVE DISTANCES
    saveDatasets = []
    for d in range(len(dataSets)):
        saveDatasets += [dataSets[d].name]
    np.save( "%sdistances_%s.npy" % (saveDir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')),
            {"predictions":predictions, "targets":targets, "paths":paths, "metrics":names, "datasets":saveDatasets})


# DIVIDE DISTANCES INTO VIRTUAL SUBGROUPS ACCORDING TO SPLITS
predictionsSplit = []
targetsSplit = []
predictionsAll = []
targetsAll = []
for d in range(len(dataSets)):
    if splits and splits[d]:
        for split in splits[d]:
            idx = np.core.defchararray.find(paths[d].astype(str), split) > 0
            predictionsSplit += [predictions[d][:,idx]]
            targetsSplit += [targets[d][:,idx]]
    if dataSets[d].name in allCorIncludes:
        predictionsAll += [predictions[d]]
        targetsAll += [targets[d]]
if predictionsSplit:
    predictions = predictionsSplit
    targets = targetsSplit


# COMPUTE CORRELATION ON SUBGROUPS
csvHeader = ["Metric"]
for d in range(len(dataSets)):
    if splits and splits[d]:
        for split in splits[d]:
            if "Mean" in mode:
                csvHeader += [ "%s%s mean" % (dataSets[d].name, split), "%s%s std" % (dataSets[d].name, split)]
            else:
                csvHeader += [ "%s%s" % (dataSets[d].name, split)]
    else:
        if "Mean" in mode:
            csvHeader += [dataSets[d].name + " mean", dataSets[d].name + " std"]
        else:
            csvHeader += [dataSets[d].name]
if "Mean" in mode:
    csvHeader += ["All mean", "All std"]
else:
    csvHeader += ["All"]
csvData = np.zeros([len(metrics), len(csvHeader)-1]) # first text column is added later

# ADD "ALL" COLUMN TO DISTANCES
predictions += [np.concatenate(predictionsAll, axis=1)]
targets += [np.concatenate(targetsAll, axis=1)]

for i in range(len(predictions)):
    stacked = np.concatenate([predictions[i], targets[i]], axis=0)
    if not "Mean" in mode:
        if mode == "pearson":
            cor = np.corrcoef(stacked)
        elif mode == "spearman":
            cor, pVal = sciStats.spearmanr(stacked.transpose((1,0)))
        csvData[...,i] = cor[len(metrics), 0:len(metrics)] if len(metrics) > 1 else cor
    else:
        stacked = np.reshape(stacked, [stacked.shape[0], -1, cutOffIndex])
        individualCor = []
        for j in range(stacked.shape[1]):
            if mode == "pearsonMean":
                cor = np.corrcoef(stacked[:,j,:])
            elif mode == "spearmanMean":
                cor, pVal = sciStats.spearmanr(stacked[:,j,:].transpose((1,0)))
            individualCor += [cor]
        stacked = np.stack(individualCor, axis=0)
        corMean = np.mean(stacked, axis=0)
        corStd = np.std(stacked, axis=0)

        csvData[...,2*i] = corMean[len(metrics), 0:len(metrics)]
        csvData[...,2*i+1] = corStd[len(metrics), 0:len(metrics)]

# WRITE TO CSV
csvPath = saveDir + ("correlation_%s.csv" % datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
file = open(csvPath, "w")
writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
writer.writerow(csvHeader)
for m in range(len(metrics)):
    writer.writerow( [names[m]] + np.round(csvData[m], 3).tolist() )
print("Results written to %s" % csvPath)