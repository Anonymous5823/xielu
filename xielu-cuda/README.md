# Installation

make sure the `CUDA_HOME` environment variable points to the cuda toolkit directory (e.g `/usr/local/cuda`).

for linux

```pip install .```

or

```pip install . --no-build-isolation --no-deps```

for GH200 on top of existing container/uenv (cuda release wheels don't exist for aarch64, so need pip to build in isolation).
