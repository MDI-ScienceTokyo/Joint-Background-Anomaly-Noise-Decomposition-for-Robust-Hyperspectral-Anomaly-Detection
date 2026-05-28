# Joint Background-Anomaly-Noise Decomposition for Robust Hyperspectral Anomaly Detection via Constrained Convex Optimization

This is a demo code of the method proposed in the following reference:

K. Sato and S. Ono, "Joint Background-Anomaly-Noise Decomposition for Robust Hyperspectral Anomaly Detection via Constrained Convex Optimization," IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2026.

For more information, see the following

- Preprint paper : https://arxiv.org/abs/2401.14814


## How to use

### 1. Download the dataset

Please download `abu-beach-4.mat` from [Hyperspectral Datasets](https://xudongkang.weebly.com/data-sets.html) and place it in the `datasets` folder.

The directory structure should be as follows:

```text
.
├── datasets
│   └── abu-beach-4.mat
├── functions
│   ├── prox_operator
│   ├── func_JBAND_HTV.m
│   ├── func_JBAND_SSTV.m
│   ├── func_JBAND_HSSTV.m
│   ├── func_JBAND_Nuclear.m
│   └── make_noisy_data.m
└── main.m
```

### 2. Parameter settings

The main parameters can be adjusted in `main.m`.

- `use_GPU`: option for GPU acceleration. Set `use_GPU = 1` to use GPU and `use_GPU = 0` otherwise.
- `sigma`: standard deviation of Gaussian noise.
- `Sp_rate`: ratio of salt-and-pepper noise.
- `Sl_rate`: ratio of stripe noise.
- `type_DBCF`: design of the background characterization function. Available options are `'HTV'`, `'SSTV'`, `'HSSTV'`, and `'Nuclear'`.
- `lambda1`: regularization parameter for the anomaly part.
- `lambda2`: regularization parameter for the stripe noise component.

Please adjust these parameters depending on the dataset and noise level.  
For detailed parameter settings, please refer to our paper.

### 3. Run the demo

Run the following MATLAB script:

```matlab
run main.m
```

## Reference
If you use this code, please cite the following paper:
```bibtex
@misc{sato2024robust,
  doi = {10.48550/ARXIV.2401.14814},
  url = {https://arxiv.org/abs/2401.14814},
  author={Sato, Koyo and Ono, Shunsuke},
  title = {Joint Background-Anomaly-Noise Decomposition for Robust Hyperspectral Anomaly Detection via Constrained Convex Optimization},
  publisher = {arXiv},
  year = {2024},
  howpublished = {\textit{arXiv:2401.14814}}
}
```
