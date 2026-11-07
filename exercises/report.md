# Báo cáo bài 1.1: Pipeline KD → Pruning → Quantization trên CIFAR-10

Ngày chạy: 06/11/2026 · Notebook: [1.1_pipeline_cifar10.ipynb](1.1_pipeline_cifar10.ipynb)

## Tóm tắt

Ghép ba kỹ thuật của notebook 01–03 thành một pipeline, nén ResNet-18 xuống một CNN nhỏ:

- Model cuối nhỏ hơn teacher **144.4x** (44.77 MB → 0.31 MB). Accuracy giảm **1.69 điểm**
  (91.20% → 89.51%). **Đạt cả hai chỉ tiêu** (≥ 10x và ≤ 2%).
- CPU latency giảm **8x** (20.45 → 2.55 ms/ảnh). Phần giảm này đến từ KD (đổi kiến trúc) và
  pruning, không đến từ INT8.
- Bước đắt nhất là pruning: mất 1.89 điểm sau khi fine-tune. KD và INT8 gần như miễn phí.
- Biên accuracy chỉ còn **0.31 điểm**, và đây là kết quả của **một lần chạy với một seed**.

## Thiết lập

| Mục | Giá trị |
|---|---|
| Phần cứng | Google Colab, GPU Tesla T4. Latency đo trên CPU của Colab, batch 1 |
| Dữ liệu | CIFAR-10: 50k train / 10k test, random crop + horizontal flip |
| Teacher | ResNet-18, stem CIFAR (conv 3×3, bỏ maxpool), 11.17M params |
| Student | CNN kiểu VGG 6 conv, widths (64, 64, 128, 128, 256, 256), 1.15M params |
| Huấn luyện | SGD Nesterov + OneCycle, LR 0.1, batch 256, mixed precision |
| Epoch | teacher 15 · student 20 (×2) · fine-tune 5 (LR 0.02) |
| KD | T = 4, α = 0.7 |
| Pruning | L1-norm filter pruning 50% mỗi lớp conv, fine-tune bằng KD loss |
| Quantization | PTQ weight-only, symmetric, per-channel, INT8 |

Chỉ tiêu (so với teacher): nhỏ hơn ≥ 10x và accuracy giảm ≤ 2 điểm.

## Kết quả

| Bước | Params | Size | Nhỏ hơn | Acc | Δacc | CPU lat |
|---|---:|---:|---:|---:|---:|---:|
| 0. teacher ResNet-18 | 11,173,962 | 44.77 MB | 1.0x | 91.20% | +0.00 | 20.45 ms |
| 1a. student, no KD | 1,148,874 | 4.62 MB | 9.7x | 90.63% | −0.57 | 5.68 ms |
| 1b. student, KD | 1,148,874 | 4.62 MB | 9.7x | 91.38% | +0.18 | 5.61 ms |
| 2a. pruned (chưa fine-tune) | 288,746 | 1.17 MB | 38.2x | 14.38% | −76.82 | 2.57 ms |
| 2b. pruned + KD fine-tune | 288,746 | 1.17 MB | 38.2x | 89.49% | −1.71 | 2.45 ms |
| **3. + PTQ INT8** | **288,746** | **0.31 MB** | **144.4x** | **89.51%** | **−1.69** | **2.55 ms** |

| Chỉ tiêu | Kết quả | |
|---|---|---|
| Nhỏ hơn ≥ 10x | 144.4x | **PASS** |
| Accuracy giảm ≤ 2 điểm | 1.69 điểm | **PASS** |

Giảm bit-width thêm trên model đã cắt (per-channel):

| Bit | INT8 | INT6 | INT4 | INT3 |
|---|---:|---:|---:|---:|
| Accuracy | 89.51% | 89.36% | 85.85% | 79.26% |

Diễn biến huấn luyện (test accuracy):

| Model | Epoch 1 | Giữa | Cuối | Thời gian / epoch |
|---|---:|---:|---:|---:|
| Teacher (15 ep) | 48.39% | 79.07% (ep 7) | 91.20% | ~25 s |
| Student no KD (20 ep) | 57.05% | 83.15% (ep 10) | 90.63% | ~20 s |
| Student KD (20 ep) | 61.29% | 88.80% (ep 10) | 91.38% | ~22 s |
| Fine-tune KD sau khi cắt (5 ep) | 82.98% | 86.33% (ep 3) | 89.49% | ~21 s |

## Nhận xét

**Bước đắt nhất là pruning.** Cắt 50% filter làm accuracy rơi xuống 14.38%, gần mức đoán ngẫu
nhiên. Fine-tune bằng KD kéo lại lên 89.49% sau 5 epoch (+75.11 điểm), riêng epoch đầu đã đạt
82.98%. So với student chưa cắt, model vẫn mất 1.89 điểm, đổi lại params giảm 4x và latency
giảm 2.3x. Đến epoch cuối accuracy vẫn đang tăng, nghĩa là fine-tune chưa bão hoà.

**KD và INT8 gần như miễn phí.** Cùng kiến trúc, cùng số epoch, student có KD hơn student không
KD **0.75 điểm** (91.38% so với 90.63%), thậm chí vượt teacher 0.18 điểm. KD cũng giúp học nhanh
hơn ở đầu: epoch 2 đạt 74.81% so với 60.02%. INT8 không mất gì (+0.02 điểm, nằm trong nhiễu) mà
size giảm thêm 3.8x. INT6 vẫn ổn, nhưng INT4 mất 3.6 điểm, sẽ trượt chỉ tiêu.

**Latency giảm nhờ pruning, không nhờ INT8.** Structured pruning bỏ hẳn kênh, nên mạng thật sự
ít FLOPs hơn. PTQ ở đây chỉ là weight-only fake-quant: weight được dequantize ngược về FP32 và
chạy bằng đúng kernel FP32 như trước, nên latency không đổi (2.45 so với 2.55 ms là nhiễu đo).
Muốn INT8 chạy nhanh hơn thì phải quantize cả activation và chạy bằng kernel số nguyên thật
(`torch.ao.quantization` với fbgemm/qnnpack, ONNX Runtime, TensorRT).

## Hạn chế và hướng tiếp theo

- **Biên accuracy mỏng** (0.31 điểm) và mới có một seed. Cần chạy lại vài seed để biết phương sai.
- **Size đang dư rất xa chỉ tiêu** (144x so với 10x). Có thể giảm `PRUNE_RATIO` xuống 0.3–0.4
  hoặc tăng `FINETUNE_EPOCHS` lên khoảng 10 để lấy thêm biên accuracy.
- **Teacher mới train 15 epoch** (91.20%). Teacher mạnh hơn sẽ làm chỉ tiêu "so với teacher"
  khó đạt hơn, nhưng cũng có thể giúp KD tốt hơn.
- **Latency của INT8:** chưa đo được tốc độ INT8 thật vì chưa dùng kernel số nguyên.
- **Thời gian tải data:** server gốc của CIFAR-10 chỉ tải được khoảng 120 kB/s, mất 23 phút,
  còn phần train chỉ khoảng 25 phút. Lần sau nên tải từ một mirror nhanh hơn hoặc cache trên Drive.

## Tái hiện

Mở notebook bằng nút "Open in Colab", chọn `Runtime → Change runtime type → T4 GPU`, rồi
`Runtime → Run all`.

---

# Báo cáo bài 1.2: Tìm "vách đá" của quantization

Ngày chạy: 07/11/2026 (chạy lại cùng bài 1.3; lần đầu 06/11 cho cùng kết luận) · Notebook: [1.2_quant_cliff.ipynb](1.2_quant_cliff.ipynb)

## Tóm tắt

- **PTQ per-tensor** giữ được đến 5 bit. Từ **4 bit** giảm quá 1 điểm (−1.67), từ **3 bit** giảm quá 5 điểm (−46.25).
- **PTQ per-channel** lùi vách đá đi một bit. 4 bit chỉ mất 0.80 điểm, từ **3 bit** giảm quá 1 điểm (−4.83),
  đến **2 bit** thì sụp (−80.05).
- **QAT INT4 bằng STE đạt 91.38%**, cao hơn FP32 0.13 điểm. **Chỉ tiêu ≤ 1 điểm: PASS.**
- Bonus: QAT kéo INT3 về 91.27% (+0.02) và INT2 từ 11.20% lên 89.14% (−2.11, chưa đạt ngưỡng 1 điểm).

## Thiết lập

| Mục | Giá trị |
|---|---|
| Phần cứng | Google Colab, GPU Tesla T4 |
| Model FP32 | Student `Net` của bài 1.1 (VGG 6 conv, 1.15M params), train 20 epoch **có KD** từ ResNet-18 (15 epoch, 91.43%), T = 4, α = 0.7. Chưa pruning |
| PTQ | Weight-only fake-quant, symmetric, mọi `Conv2d` và `Linear`; bits 8 → 2; per-tensor và per-channel (theo output channel). BN và activation giữ FP32 |
| QAT | Parametrization `w + (deq(q(w)) − w).detach()` (STE), per-channel, fine-tune 8 epoch từ checkpoint FP32, SGD + OneCycle, LR 0.01, KD loss |
| Đo QAT | Bỏ parametrization, đưa weight ẩn qua đúng `quantize_model` của PTQ rồi đo accuracy |

## Kết quả

PTQ theo số bit (FP32 = 91.25%):

| Bit | Per-tensor | Δ | Per-channel | Δ |
|---:|---:|---:|---:|---:|
| 8 | 91.29% | +0.04 | 91.26% | +0.01 |
| 7 | 91.30% | +0.05 | 91.24% | −0.01 |
| 6 | 91.17% | −0.08 | 91.12% | −0.13 |
| 5 | 90.80% | −0.45 | 91.04% | −0.21 |
| 4 | **89.58%** | **−1.67** | 90.45% | −0.80 |
| 3 | **45.00%** | **−46.25** | **86.42%** | **−4.83** |
| 2 | 10.00% | −81.25 | **11.20%** | **−80.05** |

| Vách đá | Giảm > 1 điểm từ | Giảm > 5 điểm từ |
|---|---:|---:|
| Per-tensor | 4 bit | 3 bit |
| Per-channel | 3 bit | 2 bit |

PTQ so với QAT:

| Bit | PTQ per-tensor | PTQ per-channel | QAT per-channel | Δ QAT vs FP32 |
|---:|---:|---:|---:|---:|
| 4 | 89.58% | 90.45% | **91.38%** | **+0.13** |
| 3 | 45.00% | 86.42% | 91.27% | +0.02 |
| 2 | 10.00% | 11.20% | 89.14% | −2.11 |

| Chỉ tiêu | Kết quả | |
|---|---|---|
| QAT INT4 mất ≤ 1 điểm so với FP32 | +0.13 điểm (không mất) | **PASS** |

Lần chạy đầu (06/11) để đối chiếu: FP32 91.34%, PTQ INT4 per-tensor / per-channel 88.85% / 90.41%,
QAT INT4 / INT3 / INT2 91.45% / 91.32% / 89.40%. Vị trí vách đá và kết luận PASS giống hệt.

## Nhận xét

**Vách đá rơi đúng ở chỗ lưới quá thưa so với độ chênh giữa các filter.** Từ 8 xuống 5 bit gần như không mất
gì. Với per-tensor, một scale cho cả lớp bị filter có weight lớn nhất quyết định, nên filter nhỏ chỉ dùng được vài
mức; vì vậy per-tensor gãy sớm hơn per-channel một bit. Khoảng cách lớn nhất ở 3 bit: 86.42% so với 45.00%
(41.42 điểm). Ở 2 bit, lưới symmetric chỉ có 4 mức `{-2, -1, 0, 1} × max|w|`, mọi weight nhỏ hơn nửa max bị làm
tròn về 0, nên cả hai cách đều về mức đoán ngẫu nhiên.

**QAT xoá vách đá ở 4 và 3 bit.** PTQ INT4 per-channel thực ra cũng lọt ngưỡng (−0.80) nhưng chỉ dư 0.20 điểm.
QAT kéo lên 91.38%, ngay epoch 1 đã 90.89%. INT3 từ 86.42% lên 91.27%. INT2 hồi phục gần 78 điểm lên 89.14%, vẫn
đang tăng ở epoch cuối (86.85 → 89.14), nên thêm epoch nhiều khả năng sẽ còn lên. Accuracy của model INT-n xuất ra
trùng khớp accuracy fake-quant lúc train.

**Phải đọc +0.13 của QAT INT4 cẩn thận.** QAT là thêm 8 epoch fine-tune bằng KD, nên một phần mức tăng có thể đến
từ việc train thêm chứ không phải từ quantization. So sánh hoàn toàn công bằng cần fine-tune FP32 thêm 8 epoch làm
mốc. Ngoài ra +0.13 nằm trong nhiễu của một seed.

## Hạn chế và hướng tiếp theo

- **Weight-only, chưa fold BN.** Activation vẫn FP32. Fold BN vào conv sẽ đổi phân bố weight và có thể dời vách đá.
- **Size chưa giảm thật ở 4/3/2 bit:** bản packed vẫn lưu trong `int8`. Cần nén nhiều số vào một byte. Về lý thuyết
  weight INT4 nhỏ hơn FP32 8 lần.
- **INT2:** thử QAT lâu hơn, giữ conv đầu và lớp cuối ở 8 bit, hoặc học scale (LSQ) thay vì dùng `max|w|`.
- **Một seed, GPU không hoàn toàn tất định.** Cùng code, student FP32 đạt 91.38% (bài 1.1), 91.34% (1.2 lần đầu),
  91.25% (lần này).
- **Mount Google Drive thất bại ở lần chạy này** (`ValueError`), nên checkpoint lưu ở `./ckpt` và CIFAR-10 phải tải
  lại, mất 27 phút (khoảng 104 kB/s), lâu hơn phần tính toán.

---

# Báo cáo bài 1.3: Cho INT8 thật sự chạy nhanh trên CPU

Ngày chạy: 07/11/2026 · Notebook: [1.2_quant_cliff.ipynb](1.2_quant_cliff.ipynb), mục 10–14 (dùng chung data và
student KD với bài 1.2)

## Tóm tắt

- Static PTQ INT8 thật (cả weight lẫn activation, kernel số nguyên) bằng hai cách: PyTorch `torch.ao.quantization`
  FX và ONNX Runtime QDQ.
- **Chỉ tiêu "latency giảm ≥ 2 lần so với FP32 eager": đạt một phần.** ORT INT8 đạt **3.03x ở batch 1 (PASS)** nhưng
  1.95x ở batch 64; PyTorch INT8 đạt 1.88x / 1.52x (FAIL).
- So cùng runtime, riêng INT8 chỉ cho **1.5–1.8x**. Phần còn lại của 3.03x đến từ việc ORT FP32 đã nhanh hơn
  PyTorch eager 1.95x.
- Kích thước giảm 3.9 lần. PyTorch INT8 gần như giữ nguyên accuracy (−0.09), ORT INT8 mất 1.28 điểm.

## Thiết lập

| Mục | Giá trị |
|---|---|
| Phần cứng | CPU runtime Colab: Intel Xeon @ 2.00GHz, 2 luồng, AVX2 + AVX-512F, **không có VNNI/AMX** |
| Phần mềm | torch 2.14.1 (quantized engine `x86`), onnxruntime 1.31.1 |
| Model | Student KD của bài 1.2 (FP32 91.25%) |
| Calibration | 512 ảnh train, transform của tập test |
| PyTorch | `prepare_fx` / `convert_fx`, qconfig mặc định `x86` (weight INT8 per-channel, activation UINT8) |
| ONNX Runtime | Export (BN đã fold) → `quant_pre_process` → `quantize_static` QDQ, per-channel, U8S8, calibration MinMax mặc định |
| Đo latency | Input ngẫu nhiên 3×32×32, warmup rồi median của 300 lần (batch 1) / 40 lần (batch 64); 2 luồng cho cả PyTorch và ORT |

## Kết quả

| Model | Acc | Size MB | Batch 1 ms | Tăng tốc | Batch 64 ms | Tăng tốc | ảnh/s (b64) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 (eager, mốc) | 91.25% | 4.62 | 5.627 | 1.00x | 214.35 | 1.00x | 299 |
| PyTorch FP32 (BN folded) | 91.25% | 4.60 | 5.338 | 1.05x | 211.22 | 1.01x | 303 |
| PyTorch INT8 (FX, x86) | 91.16% | 1.18 | 2.997 | 1.88x | 141.40 | 1.52x | 453 |
| ONNX Runtime FP32 | 91.25% | 4.62 | 2.887 | 1.95x | 161.31 | 1.33x | 397 |
| ONNX Runtime INT8 (QDQ) | 89.97% | 1.19 | **1.860** | **3.03x** | 109.87 | 1.95x | 583 |

Tăng tốc chỉ do INT8 (so cùng runtime):

| | Batch 1 | Batch 64 |
|---|---:|---:|
| PyTorch INT8 / PyTorch FP32 folded | 1.78x | 1.49x |
| ORT INT8 / ORT FP32 | 1.55x | 1.47x |

| Chỉ tiêu (so với FP32 eager) | Batch 1 | Batch 64 |
|---|---|---|
| PyTorch INT8 ≥ 2x | 1.88x — FAIL | 1.52x — FAIL |
| ORT INT8 ≥ 2x | **3.03x — PASS** | 1.95x — FAIL |

## Nhận xét

**INT8 thật có nhanh hơn, nhưng trên CPU này chưa tới 2x nếu so công bằng.** Khác với bài 1.1/1.2 (fake-quant,
latency không đổi), ở đây conv chạy bằng kernel số nguyên nên latency giảm thật 1.5–1.8x. Con số 3.03x của ORT
trông đẹp nhưng trộn hai hiệu ứng: đổi runtime (ORT FP32 đã 1.95x vì ít overhead và fuse sẵn) và đổi kiểu số.

**Vì sao chưa tới 2x** (giả thuyết, chưa tách riêng từng yếu tố):
- CPU không có VNNI, nên phép `u8 × s8 → s32` phải dùng chuỗi `vpmaddubsw` + `vpmaddwd` + `vpaddd` thay vì một lệnh
  `vpdpbusd`, mất phần lớn lợi thế 4x số phần tử mỗi lệnh.
- Model và ảnh nhỏ (32×32, conv đầu 3 kênh vào, 2–6 ms mỗi lần gọi), nên chi phí cố định (quantize/dequantize ở hai
  đầu, pooling, gọi kernel) chiếm phần đáng kể.
- Chỉ 2 luồng trên máy ảo dùng chung, số đo có nhiễu.

**Accuracy:** PyTorch INT8 gần như không mất (−0.09) còn ORT INT8 mất 1.28 điểm. Giả thuyết: ORT dùng calibration
MinMax (PyTorch dùng HistogramObserver, cắt bớt outlier), và U8S8 trên CPU không VNNI có thể bị bão hoà ở
`vpmaddubsw`; PyTorch tránh việc này bằng `reduce_range`.

## Hạn chế và hướng tiếp theo

- **Lấy lại accuracy cho ORT:** thử `CalibrationMethod.Percentile`/`Entropy`, `reduce_range=True` hoặc activation `QInt8`.
- **Đo trên CPU có VNNI/AMX** và với nhiều luồng hơn để xem INT8 có vượt 2x không.
- **API deprecated:** `torch.ao.quantization` FX đã báo deprecated; bản sau nên dùng `torchao` pt2e
  (`prepare_pt2e` / `convert_pt2e`).
- **Một lần đo:** nên lặp lại vài lần để có khoảng dao động của latency.

## Tái hiện

Mở `1.2_quant_cliff.ipynb` bằng nút "Open in Colab" (branch `exercise-1.3-1.4`), chọn T4 GPU, `Runtime → Run all`.
Phần 1.3 tự cài `onnx`, `onnxruntime`, `onnxscript` và đo trên CPU của chính runtime đó.

---

# Báo cáo bài 1.4: Sửa kernel Triton của notebook 05

Ngày chạy: 07/11/2026 · Notebook: [1.4_triton_kernels.ipynb](1.4_triton_kernels.ipynb)

## Tóm tắt

- Kernel matmul của notebook 05, thêm `@triton.autotune` (16 config), FP16 vào / FP32 cộng dồn, ReLU fuse vào
  epilogue, sửa công thức `pid_m` của thứ tự tile theo nhóm.
- **Trên L4: trung bình 114.0% throughput cuBLAS** ở 8 kích thước lớn, thấp nhất 101.5%. **Chỉ tiêu ≥ 70%: PASS.**
- **Trên T4: 4.5% cuBLAS (FAIL).** Mọi config đều dừng ở khoảng 1 TFLOP/s, nên con số 0.06x cũ là do GPU/Triton
  trên Turing chứ không phải do code kernel: cùng config cố định của notebook 05 trên L4 đã đạt khoảng 72% cuBLAS.
- ReLU fuse coi như miễn phí (nằm trong nhiễu), còn `torch.relu` riêng tốn thêm 2–29%. GROUP_M = 8 nhanh hơn
  GROUP_M = 1 đến 1.73x khi ma trận B lớn hơn L2.
- Softmax và layer norm theo tutorial đạt 0.93–1.22x torch, khoảng 78% băng thông bộ nhớ của L4.

## Thiết lập

| Mục | Giá trị |
|---|---|
| GPU chính | Colab **NVIDIA L4** (Ada, sm_89, 58 SM, L2 48 MB, 22 GB) |
| GPU đối chiếu | Colab **Tesla T4** (Turing, sm_75, 40 SM, L2 4 MB) |
| Phần mềm | torch 2.14.1+cu130, Triton 3.8.0, CUDA 13.0 |
| Autotune | 16 config: 8 config của tutorial Triton + 8 config `num_stages = 2` (vừa 64 KB shared memory của T4); key `(M, N, K)` |
| Mốc | cuBLAS = `torch.matmul` FP16; "nb05 cfg" = cùng kernel với tile cố định 64×64×32 |
| Đo | `triton.testing.do_bench`, median, warmup 25 ms, rep 100 ms |
| Kiểm tra đúng | So với matmul FP32 làm tròn về FP16, gồm kích thước lẻ 513×1000×777: sai khác tối đa 0.0625 (≈ 1 ULP) |

## Kết quả

Matmul FP16 trên **L4** (TFLOP/s):

| M × N × K | cuBLAS | Triton autotune | % cuBLAS | nb05 cfg | % cuBLAS | Config autotune chọn |
|---|---:|---:|---:|---:|---:|---|
| 256³ | 2.98 | 3.28 | 110.0% | 2.52 | 84.6% | 64×32×32 w2 s5 |
| 512³ | 18.72 | 15.42 | 82.4% | 11.38 | 60.8% | 32×64×32 w2 s5 |
| 1024³ | 33.83 | 48.77 | 144.2% | 41.12 | 121.6% | 128×128×32 w4 s4 |
| 2048³ | 60.13 | 69.91 | 116.3% | 50.99 | 84.8% | 128×128×32 w4 s4 |
| 3072³ | 62.36 | 70.96 | 113.8% | 43.32 | 69.5% | 128×256×64 w8 s3 |
| 4096³ | 55.68 | 65.03 | 116.8% | 39.99 | 71.8% | 128×256×64 w8 s3 |
| 6144³ | 57.10 | 59.34 | 103.9% | 31.85 | 55.8% | 128×256×64 w8 s3 |
| 8192³ | 54.85 | 55.68 | 101.5% | 31.03 | 56.6% | 128×256×64 w8 s3 |
| 4096×11008×4096 | 57.96 | 59.12 | 102.0% | 31.17 | 53.8% | 128×256×64 w8 s3 |
| 4096×4096×11008 | 47.43 | 54.03 | 113.9% | 31.16 | 65.7% | 128×256×64 w8 s3 |

Cùng notebook trên **T4** (TFLOP/s, các kích thước ≥ 1024):

| M × N × K | 1024³ | 2048³ | 3072³ | 4096³ | 6144³ | 8192³ | 4096×11008×4096 | 4096×4096×11008 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cuBLAS | 37.17 | 34.95 | 30.12 | 22.61 | 14.94 | 20.91 | 14.94 | 20.42 |
| Triton autotune | 1.22 | 1.09 | 1.01 | 0.99 | 0.94 | 0.95 | 0.98 | 0.93 |
| % cuBLAS | 3.3% | 3.1% | 3.4% | 4.4% | 6.3% | 4.5% | 6.6% | 4.6% |

| Chỉ tiêu: Triton ≥ 70% cuBLAS (trung bình các kích thước ≥ 1024) | Kết quả | |
|---|---:|---|
| L4 | 114.0% (8/8 kích thước ≥ 70%) | **PASS** |
| T4 | 4.5% (0/8) | **FAIL** |

Fused ReLU và GROUP_M trên L4 (chi phí ReLU = thời gian matmul + ReLU so với matmul trần của cùng thư viện):

| M × N × K | ReLU thêm (torch) | ReLU thêm (Triton fuse) | GROUP_M = 1 TF/s | GROUP_M = 8 TF/s | G8 / G1 |
|---|---:|---:|---:|---:|---:|
| 1024³ | 8.1% | 2.3% | 48.77 | 47.66 | 0.98x |
| 2048³ | 18.6% | −0.4% | 70.49 | 71.09 | 1.01x |
| 3072³ | 9.1% | −6.0% | 78.48 | 72.41 | 0.92x |
| 4096³ | 12.8% | 4.2% | 66.66 | 60.35 | 0.91x |
| 6144³ | 2.1% | 5.2% | 43.01 | 59.17 | **1.38x** |
| 8192³ | 9.8% | −0.2% | 39.03 | 55.93 | **1.43x** |
| 4096×11008×4096 | 5.9% | 5.4% | 33.73 | 58.42 | **1.73x** |
| 4096×4096×11008 | 4.4% | 3.4% | 48.47 | 50.67 | 1.05x |

(Ở 256³ và 512³, ReLU riêng của torch tốn thêm 26–29%, Triton 0%.)

Softmax (4096 hàng, FP32) và layer norm forward (4096 hàng, FP16) trên L4, GB/s tính một lần đọc + một lần ghi:

| Số cột | Softmax Triton | `torch.softmax` | Softmax ghép op torch | Layer norm Triton | `F.layer_norm` |
|---:|---:|---:|---:|---:|---:|
| 512 | 224 | 221 | 120 | 200 | 191 |
| 1024 | 231 | 218 | 112 | 213 | 205 |
| 2048 | 233 | 221 | 84 | 223 | 220 |
| 4096 | 234 | 230 | 62 | 228 | 228 |
| 8192 | 231 | 233 | 60 | 233 | 229 |
| 12288 / 16384 | 231 | 235 | 60 | 234 | 191 |

(Hàng cuối: softmax đo ở 12288 cột, layer norm ở 16384 cột.)

## Nhận xét

**Autotune đủ để vượt cuBLAS trên L4.** Config thắng đổi theo kích thước: tile 128×128×32 (4 warp, 4 stage) cho
1024–2048, tile 128×256×64 (8 warp, 3 stage) từ 3072 trở lên. So với config cố định của notebook 05, autotune nhanh
hơn 1.2–1.9x. Việc vượt cuBLAS 1.5–44% nên đọc thận trọng: cuBLAS trên L4 chỉ đạt 47–62 TFLOP/s ở các kích thước lớn,
có thể do heuristic chọn kernel chưa tốt cho GPU này. Cả hai thư viện đều chậm dần từ 3072 trở lên; giả thuyết là
L4 (72 W) hạ xung khi chạy lâu, chưa kiểm chứng.

**0.06x cũ là chuyện của T4, không phải của kernel.** Trên T4, mọi config (tile 32×64 đến 256×128, 2–8 warp) đều cho
khoảng 1 TFLOP/s. Trần cố định không phụ thuộc tile như vậy là dấu hiệu `tl.dot` FP16 không chạy trên tensor core
ở sm_75 với Triton 3.8. Mình chưa xác nhận trực tiếp; cách kiểm tra là tìm lệnh `mma` trong PTX của kernel đã biên
dịch.

**ReLU fuse miễn phí, GROUP_M chỉ có lợi khi ma trận vượt L2.** ReLU trong epilogue thêm −6% đến +5%, nằm trong
nhiễu đo; ReLU riêng của torch tốn một kernel và một lượt đọc/ghi `C` nữa, thêm 2–29%. GROUP_M = 8 nhanh hơn
1.38–1.73x khi B (75–134 MB) lớn hơn L2 48 MB. Khi B vừa L2 (3072–4096) thì GROUP_M = 1 lại nhanh hơn 8–9%.
Riêng 4096×4096×11008 chỉ được 1.05x dù B cũng 90 MB, mình chưa giải thích được.

**Softmax và layer norm đã gần trần băng thông.** Cả hai đạt 230–234 GB/s ở các cột lớn, khoảng 78% của 300 GB/s
theo spec L4, ngang torch. Lợi ích rõ của fuse thấy được khi so với softmax ghép từ các op torch: chậm hơn 1.8–3.9x
vì đọc/ghi tensor nhiều lần.

## Hạn chế và hướng tiếp theo

- **Mỗi GPU một lần chạy.** Hai lần chạy trên L4 lệch nhau vài phần trăm, nên chênh lệch dưới khoảng 5% là nhiễu.
- **T4:** xác nhận giả thuyết tensor core bằng PTX. Nếu đúng thì trên Turing nên dùng cuBLAS, hoặc một bản Triton
  cũ hơn còn hỗ trợ MMA cho sm_75.
- **Chưa thử:** A/B chuyển vị, batch matmul, FP8/INT8, và backward của layer norm.
- **A100/H100:** chưa chạy. Colab có hiện lựa chọn A100 nhưng lần này dùng L4.

## Tái hiện

Mở notebook bằng nút "Open in Colab" (branch `exercise-1.3-1.4`), chọn `Runtime → Change runtime type → L4 GPU`
(hoặc A100), rồi `Runtime → Run all`. Khoảng 5 phút trên L4, phần lớn là autotune.
