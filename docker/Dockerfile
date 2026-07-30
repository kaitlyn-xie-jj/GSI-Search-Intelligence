# 1. 指定官方 Unsloth 镜像作为基础
FROM unsloth/unsloth:2025.11.2-pt2.8.0-cu12.8-trans4.57.0

# 2. (可选) 安装系统级依赖
# 切换到 root 用户来运行 apt
USER root
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 3. (可选) 安装额外的 Python 库

# 切换回 unsloth 用户，确保包安装在用户环境下
USER unsloth

RUN pip install --no-cache-dir \
    tensorboard==2.20.0 \
    tenacity==9.1.2 \
    gurobipy

USER root
# 4. 设置工作目录（保持官方默认）
WORKDIR /workspace