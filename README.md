# Person Re-Identification Datasets for Federated Continual Learning

![license](https://img.shields.io/github/license/ContainerSolutions/locust_exporter.svg) 

*Person re-identification* and *federated continual learning* has drawn intensive attention in the computer science society in recent decades. As far as we know, this project collects most public datasets that have been tested by person re-identification algorithms, that could be shuffled into **different camera visual angles** and **various time sequences** to satisfied the *federated learning* and *continual learning* demands across time and space. 

- If you use any of them, **please refer to the original license**. 

- If you have any suggestions or you want to include your dataset here, **please open an issue or pull request**.

- Our Project is a ground-up re-implement of the previous version, [awesome-reid-dataset](https://github.com/NEU-Gou/awesome-reid-dataset).

# Quick Start

Federated Continual Learning applied in Re-ID exists at least 2 issues for datasets:

> 1. Various camera angles are needed for collaborative training across nodes;
> 2. Multi-sequences condition, as different tasks, require fulfilled in continual learning.

It is quite pity to say that currently, there are not exists an adequate dataset holding various camera visual angles and time sequences to represent the real Re-ID scene. In that case, we provide a solution that mixture and shuffle the current existing public datasets into various angles and sequences to represent reality.

You can quickly split the datasets with default configuration for your experiment as follows before download them:

 ```shell
 $ python3 main.py \
     --datasets prid prid2011 market1501 pku msmt17 \
     --roots ./datasets/ethz \
             ./datasets/prid_2011 \
             ./datasets/Market-1501 \
             ./datasets/pku_reid \
             ./datasets/MSMT17 \
     --output ./datasets/preprocessed \
     --split_indice 0.8 0.1 0.7 \
     --task_indice 5 10 \
     --random_seed 123 
 ```

# Datasets

|                           Dataset                            | Release time |  Identity  | Cameras | Sequences | Images  |                           Download                           |
| :----------------------------------------------------------: | :----------: | :--------: | :-----: | :-------: | :-----: | :----------------------------------------------------------: |
| [ETHZ](http://homepages.dcc.ufmg.br/~william/datasets.html)  |     2007     | 85; 35; 28 |    1    |     3     |  8,580  | [Google Drive](https://drive.google.com/file/d/1kIx_5igv16eyA7ZeCchpjkRywH2uoV2b/view?usp=sharing) |
| [PRID2011](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/) |     2011     |    934     |    2    |     1     | 24,541  | [Google Drive](https://drive.google.com/file/d/1lOZSZEehCuVSgLLNBAamXAK6b9vJTMw0/view?usp=sharing) |
| [Market1501](http://www.liangzheng.com.cn/Project/project_reid.html) |     2015     |    1501    |    6    |     6     | 32,217  | [Google Drive](https://drive.google.com/file/d/1qu9V5WLADH43f8_a6bsfe6mkYlPAdMJL/view?usp=sharing) |
| [PKU-ReID](https://github.com/charliememory/PKU-Reid-Dataset) |     2016     |    114     |    2    |     1     |  1,824  | [Google Drive](https://drive.google.com/file/d/1OI3fA4HipmgubbYjAxdFrTcku9OmJ5Kc/view?usp=sharing) |
|   [MSMT17](http://www.pkuvmc.com/publications/msmt17.html)   |     2018     |    4101    |   15    |     1     | 126,441 | [Google Drive](https://drive.google.com/file/d/1JEDDBPV8y7D7y_s6rWbz1csIUoT-getA/view?usp=sharing) |
|     [DukeMTMC-ReID](http://vision.cs.duke.edu/DukeMTMC/)     |     2017     |    1812    |    8    |     1     | 36,441  | [Google Drive](https://drive.google.com/file/d/1TFSPnSwzGmzyJ3AGCFkPMskcnyJBDlem/view?usp=sharing) |

# Contributing

Pull requests are more than welcome! If you have any questions please feel free to contact us.

E-mail: [gygao@njust.edu.cn](mailto:gygao@njust.edu.cn); [ryancheung98@163.com](mailto:RyanCheung98@163.com)

# License

Copyright 2021, MSNLAB, NUST SCE

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

