# vla.cpp - run SmolVLA

[vla.cpp](https://fai-modelopt-tech.github.io/vla-cpp.github.io/) is a portable C++
inference runtime for Vision-Language-Action (VLA) models, built on top of
[llama.cpp](https://github.com/ggml-org/llama.cpp). Each model ships as one self-contained
bundle and runs unchanged from a consumer GPU down to an 8 GB embedded module.

This app shows the optimization ideas from the examples applied to a real VLA model:
**SmolVLA**.

- Project page: https://fai-modelopt-tech.github.io/vla-cpp.github.io/
- Models: https://huggingface.co/collections/vrfai/vlacpp-model-bundles

No local Linux + NVIDIA setup? [run_smolvla_colab.ipynb](run_smolvla_colab.ipynb) runs every
step below on Google Colab.

## Build server and install sim env for client

Install the server's dependencies, then clone and build `vla.cpp` server with CUDA
(CMake fetches and patches `llama.cpp` itself):
```bash
sudo apt-get install -y libzmq3-dev cppzmq-dev libprotobuf-dev protobuf-compiler
git clone https://github.com/VinRobotics/vla.cpp.git
cd vla.cpp
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_GRAPHS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURE
cmake --build build -j$(nproc)
```

The CUDA build needs an Ampere or newer GPU (`CUDA_ARCHITECTURE` >= `80`, e.g. `86` for RTX 30-series):
its kernels use `sm_80` instructions, so on an older GPU such as Colab's T4 (`75`) `ptxas` fails.
There, build the CPU server instead with `-DGGML_CUDA=OFF`.

Create simulation env with [uv](https://github.com/astral-sh/uv):
```bash
bash eval/sim/libero/setup_libero.sh
```

If `UV_SYSTEM_PYTHON` is set (Colab sets it), `uv` installs into the system Python instead of the
script's Python 3.10 venv and LIBERO's old pins fail to build; `unset UV_SYSTEM_PYTHON` first.

## Get the SmolVLA bundle

Download the model checkpoint
```bash
huggingface-cli download vrfai/smolvla-libero-gguf --local-dir models/smolvla
```

## Run

Start the server
```bash
./build/vla-server models/smolvla/smolvla-libero.gguf
```

Run an LIBERO episode
```bash
source eval/sim/libero/libero_uv/.venv/bin/activate
python eval/client/run_sim_client_direct.py \
    --task libero_object --task-id 0 --n-episodes 1 \
    --output-dir /tmp/libero_outputs \
    --arch smolvla
```

Adding `--n-action-steps 10` runs 10 actions per model call (10x fewer calls), which is what
upstream uses for SmolVLA and keeps a CPU server practical.

Output video of the simulation can be found at `/tmp/libero_outputs`.

![video](../../assets/smolvla-libero-object.gif)