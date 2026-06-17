FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# 1. Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 \
      python3-pip \
      git \
      curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Create the "python" alias
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /exp

# 3. Setup folders & permissions
RUN mkdir -p /.local /.cache /.config && \
    chmod -R 777 /.local /.cache /.config

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV SHELL /bin/bash
CMD ["bash"]