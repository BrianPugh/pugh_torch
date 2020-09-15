#!/bin/bash
# Example script on how to build this container
docker build -t brianpugh/pugh-torch:${1:-debug} .
