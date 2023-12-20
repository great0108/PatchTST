# OrthoMixer

### This is an offical implementation of OrthoMixer

## Model Overview

OrthoMIxer is composed of transformer based encoder and orthogonal feature mixer.
Orthogonal feature mixer makes the model understand relationships between variables and learn temporal and local information separately.
The overall structure of the model can be seen in the ```pic/Model.png```

## Getting Started

1. Install requirements. ```pip install -r requirements.txt```

2. Training. You can train the model with ```PatchTST_supervised/run_longExp.py```, and you can open ```./result.txt``` to see the results once the training is done:

You can change training datasets with data loader parameter in run_longExp.py (data, root_path, data_path)

You can adjust the hyperparameters based on your needs (e.g. different patch length, different look-back windows and prediction lengths).

## Acknowledgement

We appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/yuqinie98/PatchTST

## Contact

If you have any questions or concerns, please contact us: cjh0108@hanyang.ac.kr

