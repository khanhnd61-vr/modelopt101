# Báo cáo: Triển khai mô hình tối ưu trên laptop (phần Applications)

Ngày cập nhật: 05/11/2026

## Tóm tắt

Cả hai ứng dụng của phần Applications đều chạy được trên laptop cá nhân, không cần Colab.

- **TinyML Chat** (CPU, Intel i5-11400H): chat với SmolLM2-360M bằng forward pass LLaMA tự viết.
  INT8 nhỏ hơn fp32 4 lần mà perplexity gần như không đổi. **AWQ** giảm mức tăng perplexity của
  INT4 từ +22% xuống **+5%**, kích thước vẫn 244 MB. Nhanh nhất vẫn là bf16, đạt 20,1 tok/s.
- **vla.cpp + SmolVLA** (GPU RTX 3050 Laptop 4 GB, qua WSL2): robot mô phỏng gắp thành công
  **93/100** episode `libero_object`. Giảm số bước khử nhiễu từ 10 xuống 5 làm server nhanh hơn
  **16%** (183 → 155 ms mỗi lần gọi) mà tỉ lệ thành công không giảm. Gắp nhiều vật liên tiếp trong
  một cảnh vẫn chỉ đạt 1–2/4.
- **Bài học chung:** trọng số nhỏ hơn không có nghĩa là chạy nhanh hơn. INT8/INT4 trên CPU và
  bf16/f16 trên GPU đều tiết kiệm bộ nhớ nhưng chậm hơn, vì tốc độ phụ thuộc vào kernel.

## Mục tiêu và phạm vi

Mục tiêu là đưa các ý tưởng tối ưu mô hình trong 5 notebook của modelopt101 (quantization,
distillation, pruning, NAS, kernel optimization) vào mô hình thật, chạy trên laptop cá nhân.

Yêu cầu đặt ra:

- không cần Colab, cài và chạy được trên máy cá nhân;
- đo kích thước, chất lượng và tốc độ trước và sau khi tối ưu.

Máy dùng để đo:

| Thành phần     | Cấu hình                                                |
| ---------------- | --------------------------------------------------------- |
| CPU              | Intel i5-11400H, 6 nhân, 16 GB RAM                       |
| GPU              | NVIDIA RTX 3050 Laptop, 4 GB VRAM, compute capability 8.6 |
| Hệ điều hành | Windows 11 + WSL2 Ubuntu 24.04, CUDA 12.8                 |

## Ứng dụng 1: TinyML Chat ([apps/tinyml_chat](apps/tinyml_chat/))

INT8 làm mô hình nhỏ hơn 4 lần mà không mất chất lượng, và AWQ giúp INT4 gần bằng chất lượng fp32
(perplexity +5%). Tuy vậy, cả INT8 lẫn INT4 đều chậm hơn bf16, vì kernel số nguyên của PyTorch trên
CPU chưa được tối ưu.

Ứng dụng chat với mô hình
[SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) (kiến trúc
LLaMA) trên CPU. Cách làm theo ý tưởng của
[TinyChatEngine](https://github.com/mit-han-lab/TinyChatEngine), viết bằng PyTorch thuần:

- **Forward pass LLaMA tự viết** trong [tinychat.py](apps/tinyml_chat/tinychat.py): RMSNorm, RoPE,
  grouped-query attention, SwiGLU. Không dùng code mô hình của `transformers`.
- **Lượng tử hóa trọng số** cho mọi lớp linear, theo công thức của notebook 01_quantization:
  - INT8: đối xứng, một scale cho mỗi kênh đầu ra;
  - INT4: affine, một cặp scale và zero point cho mỗi nhóm 64 trọng số;
  - AWQ4: INT4 sau khi scale theo activation (xem nhận xét bên dưới).
- **KV cache dùng lại giữa các lượt chat**, nên mỗi lượt chỉ xử lý phần token mới.
- **Kiểm tra đúng đắn** bằng [test_tinychat.py](apps/tinyml_chat/test_tinychat.py):
  - logits khớp với `LlamaForCausalLM` của transformers (sai số < 1e-4), kể cả khi chạy qua
    KV cache;
  - sau khi áp scale AWQ, logits của mô hình fp32 không đổi. Khi cố ý bỏ bước gộp `1/s`, test
    báo lỗi ngay.

Cách chạy:

```bash
cd apps/tinyml_chat
pip install -r requirements.txt
python tinychat.py                    # chat, trọng số bf16
python tinychat.py --precision awq4   # chat với trọng số 4-bit sau AWQ
python tinychat.py --bench            # kích thước / perplexity / tốc độ của mọi precision
python test_tinychat.py               # tự kiểm tra
```

Kết quả đo bằng `python tinychat.py --bench`, sinh 64 token theo cách greedy:

| Độ chính xác | Kích thước trọng số (MB) |      Perplexity | Tốc độ sinh (tok/s) |
| ---------------- | ----------------------------: | --------------: | ---------------------: |
| fp32             |                          1447 |           33,19 |                   15,4 |
| bf16             |                           724 |           33,22 |         **20,1** |
| int8             |                           363 |           33,22 |                    6,0 |
| int4             |                           244 |           40,58 |                    2,3 |
| awq4             |                           244 | **34,84** |                    2,9 |

Nhận xét:

- **Kích thước giảm chắc chắn.** INT4 nhỏ hơn fp32 khoảng 6 lần. Lớp đầu ra `lm_head` được giữ ở
  INT8, vì nếu hạ cả lớp này xuống INT4 thì perplexity lên 45,3.
- **AWQ gần như xoá được thiệt hại của INT4 mà không tốn thêm bộ nhớ.** Mỗi kênh đầu vào được nhân
  với `s = mean|x|^α` trước khi làm tròn về INT4. `α` được chọn cho từng lớp để lỗi đầu ra nhỏ
  nhất trên một đoạn văn hiệu chỉnh riêng (không phải đoạn văn dùng để đo perplexity). `1/s` được
  gộp vào RMSNorm phía trước hoặc vào `up_proj`, nên không thêm phép tính nào. Perplexity tăng +5%
  thay vì +22%, và câu trả lời mẫu đúng ý hơn hẳn bản int4.
- **Tốc độ phụ thuộc vào kernel, không chỉ vào số bit.** Sinh token bị giới hạn bởi băng thông bộ
  nhớ, nên bf16 nhanh hơn fp32 1,3 lần. INT8/INT4 đọc ít byte hơn nữa nhưng chậm hơn, vì kernel số
  nguyên tổng quát của PyTorch kém hơn GEMM float của MKL. TinyChatEngine thắng ở điểm này nhờ
  kernel INT4 viết tay dùng SIMD và đa luồng, đúng chủ đề của notebook 05_kernel_optimization.
- **Giới hạn của phép đo.** Perplexity được đo trên một đoạn văn ngắn (khoảng 100 token), nên chênh
  lệch cỡ 1 điểm có thể chỉ là nhiễu. Tốc độ lần đo này cao hơn lần trước ở mọi độ chính xác (ví dụ
  fp32 15,4 so với 9,1 tok/s), có lẽ do trạng thái nguồn điện của máy, nhưng thứ tự giữa các độ
  chính xác giữ nguyên.

## Ứng dụng 2: SmolVLA chạy bằng vla.cpp ([apps/vla.cpp](apps/vla.cpp/))

SmolVLA chạy trên GPU laptop 4 GB và gắp thành công 93/100 episode `libero_object`. Giảm số bước
khử nhiễu xuống 5 là cách tăng tốc tốt nhất: −16% độ trễ mà tỉ lệ thành công không giảm.

[vla.cpp](https://fai-modelopt-tech.github.io/vla-cpp.github.io/) là runtime C++ để suy luận mô
hình Vision-Language-Action, xây trên llama.cpp. Mô hình dùng bản đóng gói GGUF
[vrfai/smolvla-libero-gguf](https://huggingface.co/vrfai/smolvla-libero-gguf) (1,1 GB). Robot được
mô phỏng trong LIBERO, với yêu cầu gắp một vật và bỏ vào giỏ.

### Cách triển khai trên laptop

1. Build `vla-server` với CUDA trong WSL2. RTX 3050 là GPU đời Ampere (`sm_86`) nên chạy được
   kernel CUDA của vla.cpp, vốn cần từ `sm_80` trở lên.
2. Cài môi trường mô phỏng LIBERO bằng `setup_libero.sh` của vla.cpp. Phiên bản cuối cùng: torch
   2.5.1+cu124, mujoco 2.3.2, lerobot 0.4.3.
3. Viết hai file mới trong repo để chạy từ terminal VS Code, thay cho notebook Colab:
   - [run_local.sh](apps/vla.cpp/run_local.sh): khởi động server (chỉ nhận kết nối từ
     localhost), chờ server sẵn sàng, chạy client rồi tắt server. Có 3 chế độ `one`, `all`,
     `multi`, và các biến môi trường `N` (số episode mỗi task), `TASKS` (danh sách task),
     `SERVER_ARGS` (cờ cho server), `OUT` (thư mục kết quả).
   - [run_multi_pick.py](apps/vla.cpp/run_multi_pick.py): gắp nhiều vật liên tiếp trong cùng một
     cảnh. Script đổi câu lệnh cho từng vật, kiểm tra vật đã vào giỏ bằng predicate của LIBERO,
     đưa tay robot về vị trí và hướng cổ tay ban đầu sau mỗi vật, và bỏ qua vật đã bị đổ. Mọi
     bước gắp đều do SmolVLA điều khiển: script không đọc vị trí vật để điều khiển và không can
     thiệp vào vật lý mô phỏng.

```bash
wsl bash apps/vla.cpp/run_local.sh one     # 1 episode, task 0
wsl bash apps/vla.cpp/run_local.sh all     # cả 10 task libero_object
wsl bash apps/vla.cpp/run_local.sh multi   # gắp nhiều vật liên tiếp trong một cảnh
```

Khi cần đặt biến môi trường, gõ `wsl` để vào shell WSL trước (PowerShell làm hỏng dấu ngoặc kép
khi truyền qua `wsl`), rồi chạy trong đó:

```bash
# 10 episode mỗi task, server dùng 5 bước khử nhiễu
N=10 SERVER_ARGS="--num-steps 5" bash apps/vla.cpp/run_local.sh all
```

Video kết quả nằm trong `~/libero_outputs` trong WSL (từ Windows:
`\\wsl$\Ubuntu-24.04\home\<user>\libero_outputs`).

### Kết quả chế độ `all`, 10 episode mỗi task

Cấu hình mặc định, `--n-action-steps 10`: **93/100** episode thành công. Tomato sauce là task duy
nhất dưới 9/10.

| Task | Vật cần gắp    |   Thành công |
| ---- | ----------------- | -------------: |
| 0    | alphabet soup     |          10/10 |
| 1    | cream cheese      |           9/10 |
| 2    | salad dressing    |          10/10 |
| 3    | bbq sauce         |           9/10 |
| 4    | ketchup           |          10/10 |
| 5    | tomato sauce      | **6/10** |
| 6    | butter            |          10/10 |
| 7    | milk              |          10/10 |
| 8    | chocolate pudding |          10/10 |
| 9    | orange juice      |           9/10 |

### Thí nghiệm cấu hình server

Độ trễ là median của 100 lần gọi liên tiếp, đo bằng `eval/client/benchmark.py` có sẵn của vla.cpp.
VRAM là `weight_buf` do server báo. Tỉ lệ thành công tính trên cả 10 task: 10 episode mỗi task cho
cấu hình mặc định, 3 episode mỗi task cho các cấu hình còn lại.

| Cấu hình server (`SERVER_ARGS`)      | Thành công | Độ trễ mỗi lần gọi (ms) | VRAM trọng số (MiB) |
| ---------------------------------------- | -----------: | ----------------------------: | --------------------: |
| mặc định (GEMM của LM nới lên f32) | 93/100 (93%) |                           183 |                  1763 |
| `--weight-dtype bf16`                  |  27/30 (90%) |                           221 |                   985 |
| `--weight-dtype f16`                   |  27/30 (90%) |                           225 |                   985 |
| `--num-steps 5` (checkpoint dùng 10)  |  28/30 (93%) |                 **155** |                  1763 |
| `--flash-attn 1`                       |  26/30 (87%) |                           163 |                  1763 |
| `--weight-dtype bf16 --num-steps 5`    |    chưa đo |                           179 |                   985 |

### Kết quả chế độ `multi`

Cảnh của task 0, tối đa 400 bước và 1 lần thử lại cho mỗi vật. Thứ tự gắp được xếp theo những vật
đạt 10/10 khi gắp riêng, chạy 3 lần. Script bỏ qua vật đã bị đổ (nghiêng hơn khoảng 45°).

| Lần chạy | salad dressing | butter     | milk       | alphabet soup                         | Tổng |
| ---------- | -------------- | ---------- | ---------- | ------------------------------------- | ----: |
| 1          | thành công   | thất bại | thất bại | trong giỏ (bị đẩy vào tình cờ) |   2/4 |
| 2          | thành công   | thất bại | thất bại | thất bại                            |   1/4 |
| 3          | thành công   | thất bại | thất bại | bị đổ, bỏ qua                     |   1/4 |
| 4 (đưa cả hướng cổ tay về ban đầu) | thành công (154 bước) | thất bại | thất bại | thất bại |   1/4 |

Lần chạy đầu tiên, với thứ tự mặc định (alphabet soup → salad dressing → cream cheese → milk) và
chưa có bước bỏ qua vật bị đổ, cũng chỉ đạt 1/4 (salad dressing).

Ở lần chạy 1–3, bước về vị trí ban đầu chỉ đưa tay về đúng vị trí, còn hướng cổ tay giữ nguyên như
sau lần gắp trước. Giả thuyết: policy chỉ từng thấy episode bắt đầu từ một hướng cổ tay cố định,
nên cổ tay bị lệch làm hỏng các vật sau. Lần chạy 4 đưa cả hướng cổ tay về ban đầu nhưng vẫn chỉ
đạt 1/4. Video cho thấy ở các vật thất bại, tay robot lơ lửng gần các vật mà không tiến tới kẹp,
nên giả thuyết này bị loại.

### Nhận xét

- **Giảm số bước khử nhiễu là cách tăng tốc hiệu quả nhất.** Từ 10 xuống 5 bước flow-matching, độ
  trễ giảm 16% mà tỉ lệ thành công không giảm. Flash attention giảm 11%. Hai cách này chưa được thử
  kết hợp.
- **Trọng số nhỏ hơn không nhanh hơn.** bf16/f16 tiết kiệm 44% VRAM nhưng chậm hơn khoảng 20% so
  với cấu hình mặc định, vốn nới trọng số lên f32. Đây là cùng bài học với TinyML Chat: tốc độ phụ
  thuộc vào kernel. bf16 chỉ đáng dùng khi cần VRAM, ví dụ trên thiết bị nhúng.
- **Tomato sauce (task 5) là task khó nhất:** chỉ 6/10 với cấu hình mặc định.
- **Gắp nhiều vật liên tiếp vẫn khó.** Salad dressing thành công cả 3 lần, nhưng butter và milk
  thất bại cả 8/8 lần thử, dù mỗi vật đạt 10/10 ở task của chính nó. Đổi thứ tự gắp hay đưa
  hướng cổ tay về ban đầu đều không giúp được. Nhiều khả năng vì mỗi vật chỉ được huấn luyện
  trong cách bày của task riêng nó.
- **Tỉ lệ 3/3 dễ đạt nếu "gian lận", nhưng khi đó không còn đo policy.** Nếu lấy vị trí vật từ
  simulator rồi điều khiển tay bằng kịch bản viết tay, hoặc gắn vật vào tay kẹp thay vì kẹp bằng
  vật lý, thì gắp đủ 3 vật không khó. Khi đó tỉ lệ thành công phản ánh kịch bản chứ không phản ánh
  SmolVLA. Báo cáo này chỉ tính các lần mà policy tự điều khiển.
- **Đo độ trễ cần cô lập.** Thời gian do client ghi trong lúc mô phỏng bị nhiễu bởi các tiến trình
  khác: một task từng đo được 417 ms/bước và một lần timeout. Vì vậy các cấu hình được so bằng
  benchmark riêng. Hai task 4 và 7 từng đo được khoảng 46–48 ms/bước; khi chạy 10 episode mỗi
  task, cả hai đều về khoảng 31–35 ms/bước, nên đó chỉ là nhiễu.

## Kết luận và hướng tiếp theo

Cả hai ứng dụng đều cài và chạy được trên laptop cá nhân bằng một lệnh, kết quả lặp lại được. Bài
học chung: giảm số bit chắc chắn tiết kiệm bộ nhớ, còn tốc độ đến từ kernel hoặc từ giảm lượng tính
toán. AWQ cứu được chất lượng INT4, còn giảm số bước khử nhiễu làm SmolVLA nhanh hơn mà không mất
độ chính xác.

Đã làm:

- [X] Chạy 10 episode mỗi task `libero_object`: 93/100 thành công.
- [X] Đo đánh đổi tốc độ, tỉ lệ thành công và VRAM khi đổi kiểu trọng số, số bước khử nhiễu và
  flash attention.
- [X] Kiểm tra vì sao task 4 và 7 chậm: chỉ là nhiễu đo.
- [X] `multi`: sắp thứ tự gắp theo kết quả và bỏ qua vật đã bị đổ. Không cải thiện được tỉ lệ 1–2/4.
- [X] `multi`: đưa cả hướng cổ tay về ban đầu giữa các vật. Vẫn 1/4, nên hướng cổ tay không phải
  nguyên nhân.
- [X] AWQ cho INT4 của TinyML Chat: perplexity +5% thay vì +22%.

Hướng tiếp theo:

- [ ] Thử kết hợp `--num-steps 5` với `--flash-attn 1`, đo cả độ trễ lẫn tỉ lệ thành công.
- [ ] Chạy 10 episode mỗi task cho các cấu hình khác mặc định. 3 episode mỗi task chưa đủ để phân
  biệt 87% với 93%.
- [ ] `multi`: cần một policy được huấn luyện trên nhiều cách bày khác nhau (ví dụ LIBERO-90) thay
  vì chỉ đổi kịch bản gắp.
- [ ] TinyML Chat: viết kernel INT4 riêng để INT4/AWQ4 nhanh hơn bf16, và đo perplexity trên bộ dữ
  liệu dài hơn.
