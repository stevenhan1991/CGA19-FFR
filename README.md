# Flow Field Reduction via Reconstructing Vector Data from 3D Streamlines Using Deep Learning
Pytorch implementation for Flow Field Reduction via Reconstructing Vector Data from 3D Streamlines Using Deep Learning

In our code, we provide an example of recovering the five critical points vector field. The dimension of this data set is 51 by 51 by 51.

## Prerequisites
- Linux
- CUDA >= 10.0
- Python >= 3.6
- Numpy
- Pytorch = 0.4.0

## Data preparation
2n binary files are required for the model training (n is the number of traced streamlines). The first n files store the positions of sample points along the traced streamlines. The second n files store the corresponding velocities of sample points along the traced streamlines.

## Data format
Each position binary file is stored as a 1d array: [x_1,y_1,z_1,x_2,y_2,z_2,...,x_k,y_k,z_k]. Similarly, each velocity binary file is stored as: [u_1,v_1,w_1,u_2,v_2,w_2,...,u_k,v_k,w_k].


## Training models
```
cd code 
```

- Stage I training

The input is the positions and velocities of sample points along the traced streamlines. The output is a recovered low-resolution vector field.
```
python3 low_init.py 
```

- Stage II training

The input is the positions and velocities of sample points along the traced streamlines and the low-resolution vector field generated at Stage I. The output is a recovered high-resolution vector field.
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
## Acknowledgements
This research was supported in part by the U.S. National Science Foundation through grants IIS-1455886, CCF-1617735, CNS-1629914, and DUE-1833129, and the U.S. Department of Energy under contract number DE-AC02-06CH11357.
