FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

# This should be overwritten by a mapped in copy of the git repo
RUN mkdir /root/pugh_torch

# This is so the git repo doesn't have to be explicitly installed.
ENV PYTHONPATH=/root/pugh_torch/$PYTHONPATH

WORKDIR /root/pugh_torch

RUN pip install pugh-torch[all]

ENTRYPOINT /bin/bash
