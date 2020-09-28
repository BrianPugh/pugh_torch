#!/bin/bash
# Example script of how to launch a docker container to run this.

# Requires at least docker 19.03 for proper GPU support.
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

# This is the absolute path to this git repo.
DIRECTORY=$(cd `dirname $0` && pwd)

DOCKER_URI=brianpugh/pugh-torch:${1:-latest}

# For x11 forwarding
xhost +local:root

mkdir -p $HOME/.pugh_torch

# DataLoader workers need access to shared memory, so we set --ipc=host
docker \
    run -it \
    --gpus all \
    -v $HOME/.pugh_torch:/root/.pugh_torch \
    -v $DIRECTORY:/root/pugh_torch \
    --ipc=host \
    --net=host \
    ${DOCKER_URI}

xhost -local:root
