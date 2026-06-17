# swnet, [Paper](https://ieeexplore.ieee.org/document/11563717), [arxiv](https://arxiv.org/html/2604.24000v1)
Shared-Kernel Wavelet Networks for Near-Sensor Poisson Image Reconstruction, accepted by IEEE Sensors Journal
***
```text
@ARTICLE{swnet,
  author={Gong, Yuanhao and Tang, Tan and Liu, Qianyan},
  journal={IEEE Sensors Journal}, 
  title={Shared-Kernel Wavelet Networks for Near-Sensor Poisson Image Reconstruction}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/JSEN.2026.3702036}}
```
***
## 1) wavelet guided neural networks with shared-kernels
![image](PoissonNetwork.jpg)
## 2) Laplacian Fields are sparse and statistically stable.
![image](sparse.png)
## 3) train a mini network for Poisson equation
![image](acc.png)
## 4) thanks to the shared kernels, our method is mini
only 177 parameters
## 5) thanks to the data-driven, our method is more accurate
![image](rec.png)
