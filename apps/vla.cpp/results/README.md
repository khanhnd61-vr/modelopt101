# Raw results: SmolVLA on vla.cpp (RTX 3050 Laptop 4 GB, WSL2)

Suite `libero_object`, `--n-action-steps 10`. Each folder holds `vla-server.log`, `task_N.log`, and per task
`summary.txt` + `episodes.csv` (one row per episode: success, steps). `videos/` has one success (task 0) and one
failure (first failing task, usually 5 = tomato sauce) per config. `bench/` has the isolated latency benchmarks
(`benchmark.py`, 100 calls, median/p95 in `server_ms`) with the matching server logs. `progress.txt` is the run queue.

| Folder | Server flags / model | Episodes per task | Success |
| --- | --- | --: | --: |
| `A_default` | default (bf16 GGUF, LM GEMMs widened to f32) | 10 | 93/100 |
| `B_bf16` | `--weight-dtype bf16` | 3 | 27/30 |
| `C_f16` | `--weight-dtype f16` | 3 | 27/30 |
| `D_steps5` | `--num-steps 5` | 3 | 28/30 |
| `D2_steps5_N10` | `--num-steps 5` | 10 | 91/100 |
| `E_flash` | `--flash-attn 1` | 3 | 26/30 |
| `K_q8_s5` | Q8_0 GGUF + `--num-steps 5` | 10 | 85/100 |
| `L_q4_s5` | Q4_0 GGUF + `--num-steps 5` | 10 | 64/100 |
| `M_q8` | Q8_0 GGUF | 10 | 90/100 |

Q8_0/Q4_0 GGUFs come from `vla.cpp/scripts/quantize_gguf.py --type Q8_0|Q4_0` (LM weights only) and are run with
`MODEL=models/smolvla/smolvla-libero-q8_0.gguf bash apps/vla.cpp/run_local.sh all`. The GGUFs are not in the repo.
`bench/G_q8`, `H_q4`, `I_q8_s5`, `J_q4_s5` are latency-only runs of the quantized models; `F_bf16_steps5` is bf16 + `--num-steps 5`.
