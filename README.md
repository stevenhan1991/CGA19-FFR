# Flow-Field-Reduction-via-Reconstructing-Vector-Data-from-3D-Streamlines-Using-Deep-Learning
Pytorch implementation for Flow Field Reduction via Reconstructing Vector Data from 3D Streamlines Using Deep Learning


## Prerequisites
- Linux
- CUDA >= 10.0
- Python >= 3.6
- Numpy
- Pytorch = 0.4.0

## Data preparation
Binary files are required for the input. The first are the position informaton in the traced each streamline. The second are the velocity information in the traced each streamline.

## Training models
```
cd code 
```

- Stage I training
```
python3 low_init.py 
```

- Stage II training
```
python3 high_refine.py
```

## Citation 
```
@article{han2019flow,
  title={Flow field reduction via reconstructing vector data from 3{d} streamlines using deep learning},
  author={Han, Jun and Tao, Jun and Zheng, Hao and Guo, Hanqi and Chen, Danny Z and Wang, Chaoli},
  journal={IEEE Computer Graphics and Applications},
  volume={39},
  number={4},
  pages={54--67},
  year={2019}
}

```
