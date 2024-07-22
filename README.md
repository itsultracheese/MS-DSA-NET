# MS-DSA-NET

This repository contains the code and pretrained model for **"Focal Cortical Dysplasia Lesion Segmentation Using Multiscale Transformer".**


# **Use Terms**

All content within this repository, including but not limited to source code, models, algorithms, data, and documentation, are subject to applicable intellectual property laws. The rights to this project are reserved by the project's author(s) or the rightful patent holder(s). This repository's contents, protected by patent, are solely for personal learning and research purposes, and are not for commercial use. Anyone utilize this project in your research, please cite our work as follows:
> X. Zhang, Y. Zhang, C. Wang, et al., Focal Cortical Dysplasia Lesion Segmentation Using Multiscale Transformer. 2024.

# **Introduction**

Focal cortical dysplasia (FCD) is a predominant etiology of drug-resistant epilepsy, requiring surgical resection of the affected cortical regions for effective treatment. However, the accurate preoperative localization of FCD lesions in magnetic resonance(MR) images is problematic. This difficulty is attributed to the fact that structural changes indicative of FCD can be subtle or, in some cases, entirely absent. Previous attempts to automate the segmentation of FCD lesions have not yielded performance levels sufficient for clinical adoption. In this study, we introduce a novel transformer-based model, MS-DSA-NET, designed for the end-to-end segmentation of FCD lesions from multi-channel MR images. The core innovation of our proposed model is the integration of a CNN encoder with a multiscale transformer to augment the feature representation of FCD lesions. A memory- and computation-efficient dual-self-attention (DSA) module is utilized within the transformer pathway to discern interdependencies among feature positions and channels, thereby emphasizing areas and channels relevant to FCD lesions. We evaluated the performance of MS-DSA-NET through a series of experiments using both subject-level and voxel-level metrics. The results indicate that our model offers superior performance both quantitatively and qualitatively. It successfully identified FCD lesions in 82.4% of patients, with a low false-positive lesion cluster rate of 0.176(std: 0.381) per patient. Furthermore, the model achieved an average Dice coefficient of 0.410(std:0.288), outperforming five established methods. Given these outcomes, MS-DSA-NET has the potential to serve as a valuable assistive tool for physicians, enabling rapid and accurate identification of FCD lesions.

# **Model** 

Flowchart of the proposed MS-DSA-NET is displayed as following. It employs the Encoder-Decoder architecture with four parallel DSA transformer pathways connect them. DSA is designed to overcome the limitation of transformer self-attention module, e.g., heavy computation complexity. We provide the trained model for FCD lesion segmentation using preprocessed T1+FLAIR images of UHB dataset[^1], which can be downloaded from [BaiduYun](https://pan.baidu.com/s/1jJWW6kdMxCp5wqV2oQRvmw) with verification code: uoq1 or [Google Drive](https://drive.google.com/drive/folders/1jB6829Yx5B2J3DNLhuquXqvHUaDDcB3Z).

![model architecture](https://github.com/zhangxd0530/MS-DSA-NET/blob/main/model.png "model architecture")

# **Usage**

1. Clone the source code using git clone command to a local directory.
2. Download the the pretrained weights file "model.pth" from [BaiduYun](https://pan.baidu.com/s/1jJWW6kdMxCp5wqV2oQRvmw) with verification code: uoq1 or or [Google Drive](https://drive.google.com/drive/folders/1jB6829Yx5B2J3DNLhuquXqvHUaDDcB3Z).
3. Create a folder named as "pretrained" in the root directory and move the downloaded "model.pth" to it.
4. Prepare raw images files in the directory "inputs/raw". For each patient, a seperate folder named like "sub-00140" is created first; Then copy the raw T1w image (t1.nii.gz) and FLAIR image (flair.nii.gz) to it. We also prepared a sample test data (./inputs/raw/sub-00140) copied from UHB dataset for easy usage. The template file MNI152_T1_1mm.nii.gz is also uploaded and the users should download it to the folder "./inputs" for usage of following preprocessing. 
5. Run seg_fcd_test.py using python to predict the FCD lesion maps. During runing, raw images are first preprocessed using FSL and the result images are saved in "./inputs/fsl/sub-xxxxx“ (t1_reg.nii.gz, flair_reg.nii.gz along with other temporary files ). The result images (t1_reg.nii.gz, flair_reg.nii.gz) are loaded and fed into the proposed model to generate the lesion map, which would be saved in the "outputs/date/sub-xxxxx" folder with name "t1_reg_seg.nii.gz". We could display the results using MITK software as below (prediction map (red) and ground truth (yellow) are blended for rendering).

![prediction map](https://github.com/zhangxd0530/MS-DSA-NET/blob/main/prediction.png)


# **Requirements**

- Python 3.8.0
- Pytorch 1.12.0
- MONAI 1.2.0
- FSL 6.0.7

# **References**
[^1]: F. Schuch, L. Walger, M. Schmitz, et al., An open presurgery mri dataset of people with epilepsy and focal cortical dysplasia type ii, Scientific Data 10 (1) (2023) 475.
