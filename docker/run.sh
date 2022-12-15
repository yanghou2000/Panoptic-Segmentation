#!/usr/bin/env bash

IMAGE=yanghou/panoptic-segmentation

#docker run -ti \
#docker run --gpus device=2 -ti \
nvidia-docker run --gpus all -ti \
 --shm-size=1g \
 -v /Volumes/:/Volumes/:rw \
 -v $HOME/data:/root/data:rw \
 -v $HOME/project/Panoptic-Segmentation:/root/code/Panoptic-Segmentation:rw \
 --rm $IMAGE:latest /bin/bash
