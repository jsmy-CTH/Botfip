FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV TZ=Asia/Shanghai
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y wget curl bzip2 ca-certificates && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN conda install -y scipy scikit-learn matplotlib pandas seaborn networkx sympy  && \
    pip install wandb omegaconf tqdm torchquad

WORKDIR /workspace
CMD [ "/bin/bash" ]
