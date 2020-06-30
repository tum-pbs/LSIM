# Learned Simulation Metric (LSiM)
This repository contains the source code for the paper [Learning Similarity Metrics for Numerical Simulations](https://arxiv.org/abs/2002.07863) by [Georg Kohl](https://ge.in.tum.de/about/georg-kohl/), [Kiwon Um](https://ge.in.tum.de/about/kiwon/) and [Nils Thuerey](https://ge.in.tum.de/about/n-thuerey/), that will be published at [ICML 2020](https://icml.cc/Conferences/2020).

*LSiM* is a metric intended as a comparison method for dense 2D data from numerical simulations. It computes a scalar distance value from two inputs that indicate the similarity between them, where a higher value means they are more different. Simple vector space metrics like an L<sup>1</sup> or L<sup>2</sup> distance are suboptimal comparison methods, as they only consider pixel-wise comparisons and cannot capture structures on different scales or contextual information. Even though they represent improvements, the commonly used structural similarity (SSIM) and peak signal-to-noise ratio (PSNR) suffer from similar problems.

![Plume Comparison](Images/plumeComparisonPlot.png)
Instead, *LSiM* extracts deep feature maps from both inputs and compares them. This means similarity on different scales and recurring structures or patterns are considered for the computation. This can be seen in the figure above, where two smoke plumes with different amounts of noise added to the simulation are compared to a reference without any noise. The normalized distances (lower means more similar) on the right indicate that *LSiM* (green) correctly recovers the ground truth change (GT, grey), while L<sup>2</sup> (red) yields a reversed ordering. Furthermore, *LSiM* guarantees the mathematical properties of a pseudo-metric, i.e., given any three data points *A*, *B*, and *C* the following holds:
- **Non-negativity:** Every computed distance value is in [0,&infin;]
- **Symmetry:** The distance *A*&rarr;*B* is identical to the distance *B*&rarr;*A*
- **Triangle inequality:** The distance *A*&rarr;*B* is shorter or equal to the distance via a detour (first *A*&rarr;*C*, then *C*&rarr;*B*)
- **Identity of indiscernibles (relaxed):** If *A* and *B* are identical the resulting distance has to be 0

Further information is available at our [project website](https://ge.in.tum.de/publications/2020-lsim-kohl/).

-----------------------------------------------------------------------------------------------------

## Installation
In the following, Linux is assumed as OS but the installation on Windows should be similar. First, clone this repository to a destination of your choice.
```
git clone https://github.com/tum-pbs/LSIM
cd LSIM
```
We recommend to install the required python packages (see `requirements.txt`) via a conda environment (using [miniconda](https://docs.conda.io/en/latest/miniconda.html)), but it may be possible to install them with *pip* (and e.g. *venv* for separate environments) as well.
```
conda create --name LSIM_Env --file requirements.txt
conda activate LSIM_Env
```

## Basic Usage
To evaluate the metric on two numpy arrays `arr1, arr2` you only need to load the model and call the `computeDistance` method. Supported input shapes are `[width, height, channels]` or `[batch, width, height, channels]`, with one or three channels.
```python
from LSIM.distance_model import *
model = DistanceModel(baseType="lsim", isTrain=False, useGPU=True)
model.load("Models/LSiM.pth")
dist = model.computeDistance(arr1, arr2)
```
The inputs are automatically normalized and interpolated to the standard network input size of `224x224` by default. Since the model is fully convolutional different input shapes are possible, and we determined that the metric still works well for spatial input dimensions between `128x128 - 512x512`. Outside this range the model performance can drop significantly, and too small inputs can cause issues as the feature extractor needs to reduce the input dimensions.
The input processing can be modified via the optional parameters `interpolateTo` and `interpolateOrder`, that determine the resulting shape and interpolation order *(0=nearest, 1=linear, 2=cubic)*. Set both to `None` to disable the interpolation.

The resulting numpy array `dist` contains distance values with shape `[1]` or `[batch]` depending on the shape of the inputs. If the evaluation should only use the CPU, set `useGPU=False`. A more detailed example is shown in `distance_example.py`; to run it use:
```
python Source/distance_example.py
```
If you encounter problems with installing, training, or evaluating the metric, let us know by opening an [issue](https://github.com/tum-pbs/LSIM/issues).

-----------------------------------------------------------------------------------------------------

## Data Download
**[Final storage of the data sets is still in progress, but a download link will be coming soon]**

## Metric Comparison
To compare the performance of different metrics on the data, use the metric evaluation in `eval_metrics.py`:
```
python Source/eval_metrics.py
```
It loads multiple metrics in inference mode to compute distances on different data sets, and then evaluates them via the Pearson correlation coefficient or Spearman's ranking correlation (see `mode` option). The script computes and stores the resulting distances as a numpy array and the corresponding correlation values as a CSV file in the Results directory. The stored distances can be reused for different final evaluations via the `loadFile` option. Running the metric evaluation without changes should result in values similar to Table 1 in the paper (small deviations due to minor changes in the evaluation are expected, and the optical flow experiment is not included due to compatibility issues).

## Re-training the Model
The necessary steps to re-train the metric from scratch can be found in `training.py`:
```
python Source/training.py
```
Running the training script without changes should result in a model with a performance close to our final *LSiM* metric (when evaluated with the metric evaluation discussed above). But of course, minor deviations due to the random nature of the model initialization and training procedure may cause performance fluctuations.

## Backpropagation through the Metric
Backpropagation, e.g., in the context of training a GAN is straightforward by integrating the `DistanceModel` class that derives from `torch.nn.Module` in a new network. Load the trained model weights from the Models directory with the `load` method in `DistanceModel` on initialization (see Basic Usage above), and freeze all trainable weights of the metric if required. In this case, the `forward` method of the metric should be used instead of `computeDistance` to perform the comparison operation.

-----------------------------------------------------------------------------------------------------

## Acknowledgments
This repository also contains the image-based LPIPS metric from the [perceptual similarity](https://github.com/richzhang/PerceptualSimilarity) repository for comparison.
