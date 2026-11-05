# vla.cpp - run SmolVLA

[vla.cpp](https://fai-modelopt-tech.github.io/vla-cpp.github.io/) is a portable C++
inference runtime for Vision-Language-Action (VLA) models, built on top of
[llama.cpp](https://github.com/ggml-org/llama.cpp). Each model ships as one self-contained
bundle and runs unchanged from a consumer GPU down to an 8 GB embedded module.

This app shows the optimization ideas from the examples applied to a real VLA model:
**SmolVLA**.

- Project page: https://fai-modelopt-tech.github.io/vla-cpp.github.io/
- Models: https://huggingface.co/collections/vrfai/vlacpp-model-bundles

On Windows, do every step below inside WSL2 (Ubuntu + NVIDIA driver; CUDA 12 toolkit in WSL),
cloning `vla.cpp` to `~/vla.cpp`. Then, from the VS Code terminal:
```bash
wsl bash apps/vla.cpp/run_local.sh one     # one episode, task 0
wsl bash apps/vla.cpp/run_local.sh all     # all 10 libero_object tasks
wsl bash apps/vla.cpp/run_local.sh multi   # pick 3-4 objects in a row in one scene
```
[run_local.sh](run_local.sh) starts the server, runs the client and stops the server;
[run_multi_pick.py](run_multi_pick.py) is the multi-object client. Videos go to `~/libero_outputs`
(from Windows: `\\wsl$\Ubuntu-24.04\home\<user>\libero_outputs`).
Env vars: `N` episodes per task, `TASKS` task ids, `SERVER_ARGS` extra `vla-server` flags, `OUT`.

## Results

RTX 3050 Laptop GPU (4 GB, WSL2), `--n-action-steps 10`. Success over `libero_object` episodes
(10 per task for the default, 3 per task otherwise); latency = median of 100 back-to-back calls
with vla.cpp's `eval/client/benchmark.py`; weights = `weight_buf` the server reports.

| `SERVER_ARGS`                    | success        | server latency | weights  |
|----------------------------------|---------------:|---------------:|---------:|
| (default: LM GEMMs widened to f32) | 93/100 (93%) | 183 ms | 1763 MiB |
| `--weight-dtype bf16`            | 27/30 (90%)    | 221 ms         |  985 MiB |
| `--weight-dtype f16`             | 27/30 (90%)    | 225 ms         |  985 MiB |
| `--num-steps 5` (checkpoint: 10) | 28/30 (93%)    | **155 ms**     | 1763 MiB |
| `--flash-attn 1`                 | 26/30 (87%)    | 163 ms         | 1763 MiB |

- **Fewer flow-matching steps is the speed win**: half the denoising steps, -16% latency, no
  measurable loss in success. Flash attention gives -11%.
- **Smaller weights are not faster here**: bf16/f16 save 44% of weight memory but run ~20% slower
  than the f32-widened default - the same lesson as TinyML Chat: speed depends on the kernel.
- Tomato sauce (task 5) is the hard task: 6/10 with the default config.
- `multi` (task 0 scene, salad dressing -> butter -> milk -> alphabet soup, 4 runs): 1-2 of 4
  placed. Salad dressing works every time; butter and milk fail all 8 attempts although each scores
  10/10 in its own task - likely because each was only ever trained in its own task's layout. `run_multi_pick.py` skips objects knocked over by more than ~45 degrees.
  Returning the wrist to its start orientation between objects (not only the position) did not
  change this: 1/4 again, with the arm hovering near the objects without grasping, so the wrist
  pose left by the previous pick is not the cause.

No local Linux + NVIDIA setup? [run_smolvla_colab.ipynb](run_smolvla_colab.ipynb) runs the same on
Google Colab.

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