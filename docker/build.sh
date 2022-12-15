#!/bin/bash

IMAGE=yanghou/panoptic-segmentation

export DOCKER_BUILDKIT=1
docker build -t $IMAGE:v1.0.1 -f docker/Dockerfile . \
    && docker tag $IMAGE:v1.0.1 $IMAGE:latest \
    && echo BUILD SUCCESSFUL