# S2D-MSC
This is the official implementation of the paper "S2D-MSC: Reference-Guided Weakly Supervised Sparse-to-Dense System Matrix Calibration for Multi-Patch Magnetic Particle Imaging".

## Important Dependencies:
- Python == 3.10.19
- PyTorch == 2.1.0
- NumPy == 1.26.4

## Data Preprocessing
You should first download the raw .mdf data from the openMPI website: 
https://magneticparticleimaging.github.io/OpenMPIData.jl/latest/index.html

Please download the Calibration Measurements 8, 9 and 10, and put them in the OpenMPI_SM/ folder.

Make sure the following file structure:

```text
OpenMPI_SM/
├── OpenData_8.mdf
├── OpenData_9.mdf
└── OpenData_10.mdf
```

Then you can run the following command to preprocess the data:

- python data_process.py

## Obtain pseudo data
you should obtained pseudo data use a single patch SM calibration model, for example :

```text
TranSMS : https://github.com/icon-lab/ TranSMS
MKD-SM : https://github.com/guo0-0-0/MKD-SM
```

## Train
After data preprocessing, you can run the following command to train the model:

- first step: python stage1_train.py
- second step: python stage2_train.py

## Predict
After training, you can run the following command to predict the system matrix:

- python evaluate.py

## Reference
If you take advantage of this paper in your research, please cite the following in your manuscript:

