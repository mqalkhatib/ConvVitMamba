# ConvVitMamba
Source code for **ConvViTMamba: Efficient Multiscale Convolution, Transformer, and Mamba-Based Sequence modelling for Hyperspectral Image Classification** Accepted for publication in **International Journal of Remote Sensing**

The paper can be accessed through: https://www.tandfonline.com/doi/abs/10.1080/01431161.2026.2663567 (PrePrint: https://arxiv.org/abs/2604.18856)

<img width="471" height="221" alt="image" src="https://github.com/user-attachments/assets/d2345b5a-00d0-4f37-a433-f1e2426a4517" />
<img width="382" height="202" alt="image" src="https://github.com/user-attachments/assets/3119ca66-9dd5-473b-9bd7-b923354d66a9" />
<img width="492" height="236" alt="image" src="https://github.com/user-attachments/assets/4845aec3-a63d-4e25-b63f-130f5d5a901f" />
<img width="532" height="162" alt="image" src="https://github.com/user-attachments/assets/d64d5b93-fb36-449e-bad7-ec8281087f06" />

# Datasets
The proposed method was evaluated on four widely used hyperspectral datasets, namely (a) Houston, (b) QUH Pingan, (c) QUH Qingyun, and (d) QUH Tangdaowan.

# Requirements
Python 3.9.18, Tensorflow (and Keras) 2.10.0

# Results
To quantitatively measure the proposed ConvViTMamba model, three evaluation metrics are employed to verify the effectiveness of the algorithm, Overall Accuracy (OA), Average Accuracy (AA) and Cohen's Kappa (k). Also, Each class accuracy has been reported
<img width="1282" height="601" alt="image" src="https://github.com/user-attachments/assets/7111c70c-9464-442e-9575-e0d457063374" />

Model was qualitatively evaluated by visually comparing the resulting class maps.
<img width="1201" height="537" alt="image" src="https://github.com/user-attachments/assets/7aa5cecb-cbe8-4972-9767-0a39ec192d9e" />

# Citation

@article{Alkhatib26042026,
author = {Mohammed Q. Alkhatib},
title = {ConvViTMamba: efficient multiscale convolution, Transformer, and Mamba-based sequence modelling for hyperspectral image classification},
journal = {International Journal of Remote Sensing},
volume = {0},
number = {0},
pages = {1--40},
year = {2026},
publisher = {Taylor \& Francis},
doi = {10.1080/01431161.2026.2663567}}
