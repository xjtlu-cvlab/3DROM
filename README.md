# 3D Random Occlusion and Multi-Layer Projection for Deep Multi-Camera Pedestrian Localization  
## Overview  
The code and demonstration videos of the 3DROM, as well as the Terrace ground truth data created by us and used in the [[arXiv]](http://arxiv.org/abs/2207.10895), are provided here. 

Qiu, R., Xu, M., Yan, Y., Smith, J. S., & Yang, X. (2022). 3D Random Occlusion and Multi-layer Projection for Deep Multi-camera Pedestrian Localization. In S. Avidan, G. Brostow, M. Cissé, G. M. Farinella, & T. Hassner (Eds.), Computer Vision – ECCV 2022 - 17th European Conference, Proceedings (pp. 695-710). (Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics); Vol. 13670 LNCS). Springer Science and Business Media Deutschland GmbH. https://doi.org/10.1007/978-3-031-20080-9_40



## Dependencies  
The programme uses the following libraries:  
python 3.7+  
pytorch 1.4+ & tochvision  
numpy  
matplotlib  
pillow  
opencv-python  
kornia  
matlab & matlabengine  

## Data Preparation  
The datasets need to be downloaded from their official websites before running the program.  
Wildtrack: https://www.epfl.ch/labs/cvlab/data/data-wildtrack/  
Multiviewx: https://github.com/hou-yz/MVDet/  
Terrace: https://www.epfl.ch/labs/cvlab/data-pom-index-php/ (The Terrace dataset has been placed in the Data folder.)  
By default, all datasets are put in ~/Data/. The ~/Data/ folder should look like this  
Data  
├── MultiviewX/  
│ └── ...  
└── Wildtrack/  
│ └── ...  
└── Terrace/  
└── ...  

## Training  
For training, please run the programme as follows:  
python main.py -d Wildtrack or   
python main.py -d multiviewx or  
python main.py -d terrace  

## Pre-Trained Models  
The pre-trained models can be download from the [link](https://drive.google.com/file/d/11ki6CHTMzMZKNSJh-tEMRNq18msa3ZzW/view?usp=sharing).  
The programme can be run with the pre-trained models for testing:  
python main.py -d wildtrack --resume PATH or  
python main.py -d multiviewx --resume PATH or  
python main.py -d terrace --resume PATH  
The “PATH” is the path of a pre-trained model.

## The GPU Requirements  
The RTX3090 is recommended for the training to ensure sufficient GPU memory.  


