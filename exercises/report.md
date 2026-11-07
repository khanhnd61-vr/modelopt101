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
