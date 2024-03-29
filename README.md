# MS-DSA-NET

This repository contains the code and pretrained model for **"Focal Cortical Dysplasia Lesion Segmentation Using Multiscale Dual-Self-Attention Network".**

# **Introduction**

Focal cortical dysplasia (FCD) is a predominant etiology of drug-resistant epilepsy,
requiring surgical resection of the affected cortical regions for effective treatment.
However, the accurate preoperative localization of FCD lesions in magnetic resonance
(MR) images is problematic. This difficulty is attributed to the fact that structural
changes indicative of FCD can be subtle or, in some cases, entirely absent. Previous
attempts to automate the segmentation of FCD lesions have not yielded performance
levels sufficient for clinical adoption. In this study, we introduce a novel transformer-based model, MS-DSA-NET, designed for the end-to-end segmentation of FCD lesions
from multi-channel MR images. The core innovation of our proposed model is the
integration of a CNN encoder with a multiscale transformer to augment the feature
representation of FCD lesions. A memory- and computation-efficient dual-self-attention (DSA) 
module is utilized within the transformer pathway to discern interdependencies among
feature positions and channels, thereby emphasizing areas and channels relevant to
FCD lesions. We evaluated the performance of MS-DSA-NET through a series of
experiments using both subject-level and voxel-level metrics. The results indicate that
our model offers superior performance both quantitatively and qualitatively. It
successfully identified FCD lesions in 82.4% of patients, with a low false-positive lesion
cluster rate of 0.176(std: 0.381) per patient. Furthermore, the model achieved an
average Dice coefficient of 0.410(std:0.288), outperforming five established methods.
Given these outcomes, MS-DSA-NET has the potential to serve as a valuable assistive
tool for physicians, enabling rapid and accurate identification of FCD lesions.

# **Model** 

Figure 1 shows the flowchart of the proposed MS-DSA-NET. It employs the Encoder-Decoder architecture with four parallel DSA transformer pathways connect them. DSA is designed to overcome the limitation of transformer self-attention module, e.g., heavy computation complexity.  

We provide the trained models for FCD lesion segmentation using T1+FLAIR images, which can be downloaded from BaiduYun(verification code: xxx).

![model architecture](/assets/images/model.png)

# **Requirements**

- Python 3.8.0
- Pytorch 1.12.0
- MONAI 1.2.0


