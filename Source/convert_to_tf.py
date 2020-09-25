from LSIM.distance_model import *

import numpy as np
import imageio

import tensorflow as tf
from tensorflow import keras


class LSiM_Base_Tf(keras.Model):

    def __init__(self):
        super(LSiM_Base_Tf, self).__init__()
        self.channels = [32,96,192,128,128]
        self.featureMapSize = [55,26,12,12,12]

        self.slice1 = keras.Sequential([
            keras.layers.ZeroPadding2D(padding=2),
            keras.layers.Conv2D(filters=32, kernel_size=12, strides=4),
            keras.layers.ReLU(),
        ])
        self.slice2 = keras.Sequential([
            keras.layers.MaxPooling2D(pool_size=4, strides=2),
            keras.layers.ZeroPadding2D(padding=2),
            keras.layers.Conv2D(filters=96, kernel_size=5, strides=1),
            keras.layers.ReLU(),
        ])
        self.slice3 = keras.Sequential([
            keras.layers.MaxPooling2D(pool_size=4, strides=2),
            keras.layers.ZeroPadding2D(padding=1),
            keras.layers.Conv2D(filters=192, kernel_size=3, strides=1),
            keras.layers.ReLU(),
        ])
        self.slice4 = keras.Sequential([
            keras.layers.ZeroPadding2D(padding=1),
            keras.layers.Conv2D(filters=128, kernel_size=3, strides=1),
            keras.layers.ReLU(),
        ])
        self.slice5 = keras.Sequential([
            keras.layers.ZeroPadding2D(padding=1),
            keras.layers.Conv2D(filters=128, kernel_size=3, strides=1),
            keras.layers.ReLU(),
        ])

        self.N_slices = 5


    def call(self, x):
        h = self.slice1(x)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h

        return h_relu1, h_relu2, h_relu3, h_relu4, h_relu5


class TransformsInference_Tf(object):
    def __init__(self, outputSize, order, normMin = 0, normMax = 255):
        self.normMin = normMin
        self.normMax = normMax
        self.outputSize = outputSize
        self.order = order

    def __call__(self, reference, other):
        # repeat for scalar fields
        if reference.shape[reference.ndim-1] == 1:
            reference = np.repeat(reference, 3, axis=reference.ndim-1)
        if other.shape[other.ndim-1] == 1:
            other = np.repeat(other, 3, axis=other.ndim-1)

        # resize
        if self.outputSize and (self.outputSize != reference.shape[1] or self.outputSize != reference.shape[2]):
            resultRef = np.zeros([reference.shape[0], self.outputSize, self.outputSize, reference.shape[3]])
            resultOther = np.zeros([other.shape[0], self.outputSize, self.outputSize, other.shape[3]])

            zoom1 = [1, self.outputSize / reference.shape[1], self.outputSize / reference.shape[2], 1]
            resultRef = scipy.ndimage.zoom(reference, zoom1, order=self.order)
            zoom2 = [1, self.outputSize / other.shape[1], self.outputSize / other.shape[2], 1]
            resultOther = scipy.ndimage.zoom(other, zoom2, order=self.order) 
        else:
            resultRef = reference
            resultOther = other

        # normalization
        dMin = np.minimum( np.min(resultRef, axis=(0,1,2)), np.min(resultOther, axis=(0,1,2)) )
        dMax = np.maximum( np.max(resultRef, axis=(0,1,2)), np.max(resultOther, axis=(0,1,2)) )
        if (dMin == dMax).all():
            resultRef = resultRef - dMin
            resultOther = resultOther - dMin
        elif (dMin == dMax).any():
            for i in range(dMin.shape[0]):
                if dMin[i] == dMax[i]:
                    resultRef[..., i] = resultRef[..., i] - dMin[i]
                    resultOther[..., i] = resultOther[..., i] - dMin[i]
                else:
                    resultRef[..., i] = (self.normMax - self.normMin) * ( (resultRef[..., i] - dMin[i]) / (dMax[i] - dMin[i]) ) + self.normMin
                    resultOther[..., i] = (self.normMax - self.normMin) * ( (resultOther[..., i] - dMin[i]) / (dMax[i] - dMin[i]) ) + self.normMin
        else:
            resultRef = (self.normMax - self.normMin) * ( (resultRef - dMin) / (dMax - dMin) ) + self.normMin
            resultOther = (self.normMax - self.normMin) * ( (resultOther - dMin) / (dMax - dMin) ) + self.normMin

        # toTensor
        resultRef = tf.expand_dims( tf.convert_to_tensor(resultRef, dtype="float32"), 0)
        resultOther = tf.expand_dims( tf.convert_to_tensor(resultOther, dtype="float32"), 0)

        return tf.concat([resultRef, resultOther], axis=0)



class DistanceModel_Tf(keras.Model):

    def __init__(self):
        super(DistanceModel_Tf, self).__init__()

        # create base model and calibration layers and build them
        self.basenet = LSiM_Base_Tf()
        self.basenet.build((1, 224, 224, 3))

        self.normAcc = []
        self.normM2 = []
        self.normCount = []

        self.linear = [
            self.linearLayer(self.basenet.channels[0], self.basenet.featureMapSize[0]),
            self.linearLayer(self.basenet.channels[1], self.basenet.featureMapSize[1]),
            self.linearLayer(self.basenet.channels[2], self.basenet.featureMapSize[2]),
            self.linearLayer(self.basenet.channels[3], self.basenet.featureMapSize[3]),
            self.linearLayer(self.basenet.channels[4], self.basenet.featureMapSize[4]),
        ]
        self.linear[0].build((1, 55, 55, 32))
        self.linear[1].build((1, 26, 26, 96))
        self.linear[2].build((1, 12, 12, 192))
        self.linear[3].build((1, 12, 12, 128))
        self.linear[4].build((1, 12, 12, 128))


        # initialize weights from pytorch model
        modelPytorch = DistanceModel(baseType="lsim", isTrain=False, useGPU=False)
        modelPytorch.load("Models/LSiM.pth")

        loaded = []
        for layer in modelPytorch.basenet.parameters():
            weight = layer.data.numpy()
            if weight.ndim > 1:
                weight = np.transpose(weight, axes=[3,2,1,0])
            loaded += [weight]
        self.basenet.set_weights(loaded)

        for i in range(len(self.linear)):
            weight = modelPytorch.linear[i][1].weight.data.numpy()
            weight = np.transpose(weight, axes=[3,2,1,0])
            self.linear[i].set_weights([weight])

        for i in range(len(modelPytorch.normAcc)):
            acc = modelPytorch.normAcc[i].numpy()
            self.normAcc += [tf.convert_to_tensor(acc)]
            m2 = modelPytorch.normM2[i].numpy()
            self.normM2 += [tf.convert_to_tensor(m2)]
            self.normCount += [modelPytorch.normCount[i]]



    def call(self, x):
        # x has shape [2,batch, width, height, channels]

        outBase1 = self.basenet(x[0])
        outBase2 = self.basenet(x[1])

        result = tf.constant(0, shape=[x.shape[1]], dtype="float32")

        for i in range( len(outBase1) ):
            normalized1 = self.normalizeTensor(outBase1[i], i)
            normalized2 = self.normalizeTensor(outBase2[i], i)

            diff = tf.math.square(normalized2 - normalized1)

            weightedDiff = self.linear[i](diff, training=False)

            temp = tf.squeeze( tf.math.reduce_mean(tf.math.reduce_mean(weightedDiff, axis=2), axis=1) )
            result = result + temp

        result = tf.math.sqrt(result)

        return result


    # input two 4D/3D numpy arrays in order [batch, width, height, channels] or
    # [width, height, channels] and return a distance of shape [batch] or [1]
    def computeDistance(self, input1, input2, interpolateTo=224, interpolateOrder=0):
        assert (input1.shape == input2.shape), 'Both inputs must have identical dimensions!'

        in1 = input1[None,...] if input1.ndim == 3 else input1
        in2 = input2[None,...] if input2.ndim == 3 else input2

        data_transform = TransformsInference_Tf(interpolateTo, interpolateOrder, normMin=0, normMax=255)
        inputTensor = data_transform(in1, in2)

        result = self(inputTensor)
        sess = keras.backend.get_session()
        with sess.as_default():
            return result.eval()


    # 1x1 convolution layer to scale activations channel-wise
    def linearLayer(self, channelsIn, featureMapSize):
        layer = keras.Sequential([
            keras.layers.Dropout(0.5),
            keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, use_bias=False, kernel_constraint=keras.constraints.non_neg()),
        ])
        return layer


    def normalizeTensor(self, tensorIn, layer, epsilon=1e-10):
        size = tensorIn.shape

        # rescale norm accumulators for differently sized inputs
        if size[1] != self.normAcc[layer].shape[0] or size[2] != self.normAcc[layer].shape[1]:
            temp = tf.expand_dims(tf.expand_dims(self.normAcc[layer], 0), 3)
            normAcc = tf.squeeze(tf.image.resize( tf.expand_dims(tf.expand_dims(self.normAcc[layer], 0), 3), size=(size[1], size[2]) ))
            normM2 = tf.squeeze(tf.image.resize( tf.expand_dims(tf.expand_dims(self.normM2[layer], 0), 3), size=(size[1], size[2]) ))

            mean = normAcc
            mean = tf.expand_dims(tf.expand_dims(mean, 0), 3)
            std = tf.math.sqrt( normM2 / (self.normCount[layer] - 1) )
            std = tf.expand_dims(tf.expand_dims(std, 0), 3)
        # directly use norm accumulators for input size 224x224
        else:
            mean = self.normAcc[layer]
            mean = tf.expand_dims(tf.expand_dims(mean, 0), 3)
            std = tf.math.sqrt( self.normM2[layer] / (self.normCount[layer] - 1) )
            std = tf.expand_dims(tf.expand_dims(std, 0), 3)

        normalized = (tensorIn - mean) / (std + epsilon)

        normalized2 = normalized / (tf.math.sqrt(float(size[3].value)) - 1)
        return normalized2


ref = imageio.imread("Images/plumeReference.png")[...,:3]
plumeA = imageio.imread("Images/plumeA.png")[...,:3]
plumeB = imageio.imread("Images/plumeB.png")[...,:3]

model = DistanceModel_Tf()
distA = model.computeDistance(ref, plumeA, interpolateTo=224)
distB = model.computeDistance(ref, plumeB, interpolateTo=224)

print()
print("LSiM Tensorflow  --  PlumeA: %0.4f  PlumeB: %0.4f" % (distA, distB))

# distance results should look like this:
#LSiM   --  PlumeA: 0.3892  PlumeB: 0.4422

