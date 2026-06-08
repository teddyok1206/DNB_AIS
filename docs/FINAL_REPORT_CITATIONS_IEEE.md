# Final Report Citations - IEEE Style

This file is the working citation register for the final EESRL report.

Policy:

- Keep citations tied to methods or data sources actually used in the active pipeline.
- Keep retired GAT/SAR/radar references out of the main list unless the final report explicitly discusses the discarded design path.
- When a citation is added, removed, or corrected, update this file first.
- Local PDFs under `_Readings/` are not tracked by git; only citation metadata and rationale should be tracked here.

## Current Active Pipeline Coverage

```text
VIIRS DNB full-scene GeoTIFF
AIS/bbox-derived supervision
KR EEZ sea mask
PH/DRUID/cripser H0 partitioning
sum-preserving Gaussian density target
PH-assisted CountSpatial U-Net
continuous density-map output with count-by-integral evaluation
```

## Core Citations

[1] S. D. Miller et al., "Illuminating the Capabilities of the Suomi National Polar-Orbiting Partnership (NPP) Visible Infrared Imaging Radiometer Suite (VIIRS) Day/Night Band," *Remote Sensing*, vol. 5, no. 12, pp. 6717-6766, 2013, doi: 10.3390/rs5126717. [Online]. Available: https://www.mdpi.com/2072-4292/5/12/6717

Pipeline use:

```text
VIIRS DNB sensor/background reference for nighttime visible-light observation.
```

[2] L. B. Liao, S. Weiss, S. Mills, and B. Hauss, "Suomi NPP VIIRS day-night band on-orbit performance," *Journal of Geophysical Research: Atmospheres*, vol. 118, no. 22, pp. 12705-12718, 2013, doi: 10.1002/2013JD020475. [Online]. Available: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2013JD020475

Pipeline use:

```text
DNB radiometric/geometric performance and low-light imaging characteristics.
```

[3] S. Yoon, H.-T. Lee, H.-M. Choi, and H. Yang, "Verification of Night Light Satellite Data using AIS Data," in *Proc. Korean Institute of Navigation and Port Research Joint Conference*, Jeju, Korea, 2022.

Pipeline use:

```text
Local Korea-adjacent precedent for validating VIIRS DNB nighttime lights with AIS.
```

Local file:

```text
_Readings/02_dnb_ais_maritime_vessel_detection/12_Verification of Night Light Satellite Data using AIS Data.pdf
```

[4] S. Yoon, H.-T. Lee, H.-M. Choi, M. K. Kim, J. Lee, H. J. Han, and H. Yang, "A Study on the Automation of Night Fishing Vessel Detection and Extraction Techniques of DNB Night Light Satellites Using Artificial Intelligence," in *Proc. Korea Computer Congress*, 2023.

Pipeline use:

```text
Closest local precedent for DNB nighttime fishing-vessel detection using AI and AIS-based evaluation.
```

Local file:

```text
_Readings/02_dnb_ais_maritime_vessel_detection/18_인공지능을 이용한 DNB 야간불빛위성의 야간어선탐지 자동화 추출 기법연구.pdf
```

[5] V. Lempitsky and A. Zisserman, "Learning To Count Objects in Images," in *Advances in Neural Information Processing Systems*, vol. 23, 2010. [Online]. Available: https://papers.neurips.cc/paper/4043-learning-to-count-objects-in-images

Pipeline use:

```text
Foundational density-map counting idea: object count is recovered by integrating predicted density over a region.
```

[6] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," in *Medical Image Computing and Computer-Assisted Intervention - MICCAI 2015*, LNCS 9351, pp. 234-241, 2015, doi: 10.1007/978-3-319-24574-4_28. [Online]. Available: https://lmbweb.informatik.uni-freiburg.de/Publications/2015/RFB15a

Pipeline use:

```text
Encoder-decoder convolutional image-to-image architecture basis for patch-to-density-map inference.
```

[7] Y. Li, X. Zhang, and D. Chen, "CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 1091-1100, doi: 10.1109/CVPR.2018.00120. [Online]. Available: https://openaccess.thecvf.com/content_cvpr_2018/html/Li_CSRNet_Dilated_Convolutional_CVPR_2018_paper.html

Pipeline use:

```text
Density-map regression and count-by-sum baseline from congested-object counting literature.
```

[8] Z. Ma, X. Wei, X. Hong, and Y. Gong, "Bayesian Loss for Crowd Count Estimation With Point Supervision," in *Proc. IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019, pp. 6142-6151, doi: 10.1109/ICCV.2019.00624. [Online]. Available: https://openaccess.thecvf.com/content_ICCV_2019/html/Ma_Bayesian_Loss_for_Crowd_Count_Estimation_With_Point_Supervision_ICCV_2019_paper.html

Pipeline use:

```text
Motivates supervising expected counts from point annotations without over-trusting a single hard Gaussian target.
```

[9] B. Wang, H. Liu, D. Samaras, and M. H. Nguyen, "Distribution Matching for Crowd Counting," in *Advances in Neural Information Processing Systems*, vol. 33, 2020. [Online]. Available: https://papers.nips.cc/paper/2020/hash/118bd558033a1016fcc82560c65cca5f-Abstract.html

Pipeline use:

```text
Motivates separating count/mass conservation from normalized spatial distribution matching.
```

[10] H. Idrees, M. Tayyab, K. Athrey, D. Zhang, S. Al-Maadeed, N. Rajpoot, and M. Shah, "Composition Loss for Counting, Density Map Estimation and Localization in Dense Crowds," in *Proc. European Conference on Computer Vision (ECCV)*, 2018. [Online]. Available: https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Haroon_Idrees_Composition_Loss_for_ECCV_2018_paper.php

Pipeline use:

```text
Motivates composed objectives that jointly handle count, density shape, and localization.
```

[11] S. Kaji, T. Sudo, and K. Ahara, "Cubical Ripser: Software for computing persistent homology of image and volume data," *arXiv preprint arXiv:2005.12692*, 2020. [Online]. Available: https://arxiv.org/abs/2005.12692

Pipeline use:

```text
Persistent homology backend used through cripser for 2D DNB image topology.
```

[12] R. A. Shaw, S. Fotopoulou, M. Birkinshaw, N. Maddox, and H. Stewart, "DRUID: source detection and deblending in astronomical images with persistent homology," *RAS Techniques and Instruments*, vol. 4, 2025, Art. no. rzaf006, doi: 10.1093/rasti/rzaf006. [Online]. Available: https://academic.oup.com/rasti/article/doi/10.1093/rasti/rzaf006/8043275

Pipeline use:

```text
Methodological inspiration for using persistent homology as source/cluster detection and deblending prior.
```

## Optional Background Citations

Use these only if the final report needs broader literature context.

[13] B. Lee, Y.-K. Lee, and S. W. Kim, "Analysis of Unmatched Fishing Activities Between VIIRS and Field Data (AIS and V-Pass) Around Korean Peninsula," *Ocean Science Journal*, 2024, doi: 10.1007/s12601-024-00145-2. [Online]. Available: https://colab.ws/articles/10.1007/s12601-024-00145-2

Reason to keep optional:

```text
Good Korea-specific VIIRS/AIS/V-Pass density context, but not directly used in model implementation.
```

[14] G. Zuo, J. Zhou, Y. Meng, T. Zhang, and Z. Long, "Night-Time Vessel Detection Based on Enhanced Dense Nested Attention Network," *Remote Sensing*, vol. 16, no. 6, Art. no. 1038, 2024, doi: 10.3390/rs16061038. [Online]. Available: https://www.mdpi.com/2072-4292/16/6/1038

Reason to keep optional:

```text
Useful modern VIIRS/DNB vessel-detection baseline, but current report focuses on density heatmap estimation rather than bounding-box/object detection.
```

## Not Currently Mainline

Do not cite in the main method section unless discussing design history:

```text
GAT / graph attention references
SAR/radar references
general overfitting references
arctan distribution references
```
