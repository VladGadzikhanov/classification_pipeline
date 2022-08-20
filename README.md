# classification_pipeline

This repo contains the classification pipeline forged in Alrosa's project flame

## Usage scenarios
One could use this repo for:
- usual image classification task
- multi-image classification (several photos of object -> one class output)

## What you need to run pipeline for your task
The only one crucial thing is dataset. One need to create it in the format of pandas DataFrame with following **columns**:
- *class_name*: str
- *obj_id*: Any - id of one item. Unique for single image classification. Images of the same object must have the same *obj_id* (for multi image classification)
- *img_path*: str - path to image
- *target*: int - encoding of classes

## Functionality 
With help of this repo one might succesfully complete these tasks:
- Model training for image classification
- Fast results visualization
- It is easy to analyse mistakes since default pipeline returns pandas Dataframe with classification results and there are fucntions for simple analysis
- Utilities to convert models to onnx
- Tensorboard

## Models
Code for all models are in ./models. Now presented the following architechtures:
- densenet 
- densenet_attention (with attention layer for multi-image classification)
- efficientnet
- efficientnet_attn
- mobilenet_v2
- resnet50

**To add new models**:
- Write your model (or copy code from torchvision or elsewhere)
- define method **one_shot_predict** (usually could be copied from already existing models)
- *(Optional)* if your model differs sufficiently from already existent models, it is probable that you will need to write your own collection (*./src/collection*) and trainer (*./src/train*) 

# Docker manipulations
## 0. Check cuda version
If you need cuda version different from 11.4.0 (in current Dockerfile)
change docker image to needed one, list of available images is presented here: *https://hub.docker.com/r/nvidia/cuda*
## 1. create docker image
- **If you are working on solomon just use already created docker image (go to step 2)**  
- Otherwise: In project directory run:  
```
docker build . -t classification_pipe
```  
## 2. run docker container in interactive mode
```
docker run -it --rm --ipc=host --runtime=nvidia -e LANG=C.UTF-8 -p 127.0.0.1:8888 -v **/storage** :/storage -t classification_pipe /bin/bash
```  
Port 8888 is used to run jupyter notebook, one could connect to it via solomon with port *{solomon_port}*.  
To mount folder from host machine in docker container remove *{solomon_storage}* to any host folder. For example to run example notebooks one have to specify *{solomon_storage}* as **/storage**  
To connect to Tensor Board use port *{tensorboard_port}*
**RECOMMENDATION:** use **tmux** to run docker container to prevent connection losses.

# Jupyter notebook
To launch jupyter notebook server run:  
```
jupyter notebook --ip=127.0.0.0 --allow-root
```
Since then one could copy token and use it to connect to jupyter notebook via port *{solomon_port}*

**recommendation:** it is convinient to use **Table of Contents (ToC)** in jupyter notebook. To use **ToC** you need use package nbextensions which is already install with docker image. To set it on, do the following:
1. In jupyter notebook choose tab "Nbextensions"
2. Check the box "Table of Contents (2)"

# Tensor Board

## Default settings
To use Tensor Board run:
```
tensorboard --logdir experiments/tensorboard --host 0.0.0.0
```
After that you may connect to Tensor Board via *{tensorboard_port}* 
## Custom settings
If you want ot save Tensor Board logs to another folder and read from it run:  
```
tensorboard --logdir {tb_experiments_folder} --host 0.0.0.0
```  
*{tb_experiments_folder}* is folder from which tensorboard will read experimetns data
*NOTE* you need change parameter configuration parameter **experiment_param.path_tensorboard** (from *configs/classification.yaml*) in such case
