FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

# For opencv
# https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -yqq

# For open3d
# https://askubuntu.com/questions/1219027/problem-with-shared-library-libc-so-1-on-kubuntu-18-04
RUN apt-get update && apt install libc++-dev -yqq

RUN rm -rf /var/lib/apt/lists/*

RUN pip3 install pytorch-lightning

# class-resolver is an additional requirement for torch-geometric - https://github.com/pyg-team/pytorch_geometric/issues/4195
RUN pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv class-resolver torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
# For CUDA 10.2 environments
# RUN pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv class-resolver torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu102.html
# For CPU environments
#RUN pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv class-resolver torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html

# If necessary
#COPY requirements.txt /tmp/requirements.txt
#RUN #pip3 install -r /tmp/requirements.txt
#RUN rm -rf /tmp/requirements.txt

# Install TensorBoard
RUN pip install tensorboard
# Expose the port for TensorBoard
EXPOSE 6006



WORKDIR /root/code/Panoptic-Segmentation
