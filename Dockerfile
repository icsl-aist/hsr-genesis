FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev python3-pip git \
    libgl1-mesa-glx libglib2.0-0 libegl1-mesa libgomp1 xvfb \
    libxrender1 libxrender-dev libx11-dev libxxf86vm-dev libxfixes-dev libxi-dev \
    libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
    libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 libxcb-xinerama0 \
    libvulkan1 mesa-vulkan-drivers libnvidia-gl-550 \
    && rm -rf /var/lib/apt/lists/*

# libnvidia-gl-550 ships an empty nvidia_icd.json that is normally populated
# by the NVIDIA driver post-install script.  Since the driver is not fully
# installed in this container, we write the ICD manifest manually.
RUN printf '{\n    "file_format_version" : "1.0.1",\n    "ICD": {\n        "library_path": "libGLX_nvidia.so.0",\n        "api_version" : "1.4.329"\n    }\n}\n' > /usr/share/vulkan/icd.d/nvidia_icd.json

RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN pip install --upgrade pip

RUN pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
RUN pip install genesis-world==0.4.6 pytest

WORKDIR /workspace
COPY . .
RUN PYTHONPATH=src pip install -e .

COPY scripts/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["bash"]
