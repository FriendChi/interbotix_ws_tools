# Version: 1.0.0
# Update date：2025-04-09

# 使用 NVIDIA 提供的 CUDA 12.1 开发版镜像，基于 Ubuntu 22.04
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 设置环境变量，防止交互式提示，并指定时区为日本东京
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# 设置工作目录
WORKDIR /app

# 更新包列表并安装必要的工具和依赖项
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    unzip \ 
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python3-openssl \
    git \
    tzdata \
    cmake \
    screen \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws

# 下载并编译安装 Python 3.10
RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz && \
    tar xvf Python-3.10.0.tgz && \
    cd Python-3.10.0 && \
    ./configure --enable-optimizations && \
    make -j $(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.10.0 Python-3.10.0.tgz

# 创建符号链接
RUN ln -s /usr/local/bin/python3.10 /usr/local/bin/python3 && \
    ln -s /usr/local/bin/python3.10 /usr/local/bin/python

# 安装 pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# 设置环境变量，防止 pip 在安装时出现警告
ENV PIP_ROOT_USER_ACTION=ignore

#docker build -t base_image .

#docker run --gpus '"device=0,2"' -v /mnt:/app/data --shm-size=20g -it --name chi_container1 base_image

