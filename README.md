# Learned Simulation Metric (LSiM)
This repository contains the source code for the paper [Learning Similarity Metrics for Numerical Simulations](https://arxiv.org/abs/2002.07863) by Georg Kohl, [Kiwon Um](https://ge.in.tum.de/about/kiwon/) and [Nils Thuerey](https://ge.in.tum.de/about/n-thuerey/).

LSiM is a metric intended as a comparison method for dense 2D data from numerical simulations. It computes a scalar distance value from two inputs that indicate the similarity between them, where a higher value means they are more different. Simple vector space metrics like an L<sup>1</sup> or L<sup>2</sup> distance are suboptimal comparison methods, as they only consider pixel-wise comparisons and cannot capture structures on different scales or contextual information. Even though they represent improvements, the commonly used structural similarity (SSIM) and peak signal-to-noise ratio (PSNR) suffer from similar problems.

![Plume Comparison](https://ge.in.tum.de/wp-content/uploads/2020/02/lsim-plumes-1.png)
Instead, LSiM extracts deep feature maps from both inputs and compares them. This means similarity on different scales and recurring structures or patterns are considered for the computation. This can be seen in the figure above, where two smoke plumes with different amounts of noise added to the simulation are compared to a reference without any noise. The normalized distances on the right indicate that LSiM (green) correctly recovers the ground truth change (GT, grey), while L<sup>2</sup> (red) yields a reversed ordering. Furthermore, LSiM guarantees the mathematical properties of a pseudo-metric, i.e., given any three data points *A*, *B*, and *C* the following holds:
- **Non-negativity:** Every computed distance value is in [0,&infin;]
- **Symmetry:** The distance *A*&rarr;*B* is identical to the distance *B*&rarr;*A*
- **Triangle inequality:** The distance *A*&rarr;*B* is shorter or equal to the distance via a detour (first *A*&rarr;*C*, then *C*&rarr;*B*)
- **Identity of indiscernibles (relaxed):** If *A* and *B* are identical the resulting distance has to be 0

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

## Usage
To evaluate the metric on two numpy arrays `arr1, arr2` with shape `[width, height, 3]` or shape `[batch, width, height, 3]`, you only need to load the model and call the `computeDistance` method.
```python
from lsim.distance_model import *
model = DistanceModel(baseType="lsim", dataMode="all", isTrain=False, useGPU=True)
model.load("models/LSiM.pth")
dist = model.computeDistance(arr1, arr2)
```
The inputs are automatically interpolated to the correct network input size of `224x224` where the optional `order` parameter determines the order of the interpolation *(0=nearest, 1=linear, 2=cubic)*. Now, the numpy array `dist` contains distance values with shape `[1]` or `[batch]` depending on the shape of the inputs. If the evaluation should only use the CPU, set `useGPU=False`. A more detailed example is shown in `distance_example.py`. To run it use:
```
python distance_example.py
```
If you encounter problems with installing or evaluating the metric, let us know by opening an [issue](https://github.com/tum-pbs/LSIM/issues).

## Acknowledgments
This repository also contains the image-based LPIPS metric from the [perceptual similarity](https://github.com/richzhang/PerceptualSimilarity) repository for comparison.
