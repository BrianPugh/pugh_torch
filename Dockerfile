FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

ENV DEBIAN_FRONTEND=noninteractive

# Just life imporvement cli utilities
RUN apt update && apt -y install \
    autoconf \
	git \
    autotools-dev \
    cmake \
    feh \
    ffmpeg \
    fluxbox \
    libglib2.0-0 \
    libjpeg8-dev \
    python3-tk \
    vim \
    wget \
    x11vnc \
    xvfb

# This should be overwritten by a mapped in copy of the git repo
RUN mkdir /root/pugh_torch

# This is so the git repo doesn't have to be explicitly installed.
ENV PYTHONPATH=/root/pugh_torch/$PYTHONPATH

WORKDIR /root/pugh_torch

# Install from master to always get the latest.
RUN pip install "pugh-torch[all] @ git+https://github.com/BrianPugh/pugh_torch.git"

ENV DISPLAY=:20
ENV DISPLAY_RESOLUTION=1280x1440

# Run a vnc server so we can easily view stuff like the output of matplotlib
# even if the host doesn't have an X-server.
CMD x11vnc -create -env FD_PROG=/usr/bin/fluxbox -env X11VNC_FINDDISPLAY_ALWAYS_FAILS=1 -env X11VNC_CREATE_GEOM=${1:-${DISPLAY_RESOLUTION}x16} -nopw
