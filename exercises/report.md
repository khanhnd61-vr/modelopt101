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

Ngày chạy: 06/11/2026 · Notebook: [1.2_quant_cliff.ipynb](1.2_quant_cliff.ipynb)

## Tóm tắt

- **PTQ per-tensor** giữ được đến 5 bit. Từ **4 bit** giảm quá 1 điểm (−2.49), từ **3 bit** giảm quá 5 điểm (−26.81).
- **PTQ per-channel** lùi vách đá đi một bit. 4 bit chỉ mất 0.93 điểm, từ **3 bit** giảm quá 1 điểm (−4.52),
  đến **2 bit** thì sụp (−80.12).
- **QAT INT4 bằng STE đạt 91.45%**, cao hơn FP32 0.11 điểm. **Chỉ tiêu ≤ 1 điểm: PASS.**
- Bonus: QAT kéo INT3 về 91.32% (−0.02) và INT2 từ 11.22% lên 89.40% (−1.94, chưa đạt ngưỡng 1 điểm).

## Thiết lập

| Mục | Giá trị |
|---|---|
| Phần cứng | Google Colab, GPU Tesla T4 |
| Model FP32 | Student `Net` của bài 1.1 (VGG 6 conv, 1.15M params), train 20 epoch **có KD** từ ResNet-18 (15 epoch, 91.20%), T = 4, α = 0.7. Chưa pruning |
| PTQ | Weight-only fake-quant, symmetric, mọi `Conv2d` và `Linear`; bits 8 → 2; per-tensor và per-channel (theo output channel). BN và activation giữ FP32 |
| QAT | Parametrization `w + (deq(q(w)) − w).detach()` (STE), per-channel, fine-tune 8 epoch từ checkpoint FP32, SGD + OneCycle, LR 0.01, KD loss |
| Đo QAT | Bỏ parametrization, đưa weight ẩn qua đúng `quantize_model` của PTQ rồi đo accuracy |

## Kết quả

PTQ theo số bit (FP32 = 91.34%):

| Bit | Per-tensor | Δ | Per-channel | Δ |
|---:|---:|---:|---:|---:|
| 8 | 91.33% | −0.01 | 91.33% | −0.01 |
| 7 | 91.41% | +0.07 | 91.37% | +0.03 |
| 6 | 91.12% | −0.22 | 91.26% | −0.08 |
| 5 | 90.74% | −0.60 | 91.24% | −0.10 |
| 4 | **88.85%** | **−2.49** | 90.41% | −0.93 |
| 3 | **64.53%** | **−26.81** | **86.82%** | **−4.52** |
| 2 | 10.00% | −81.34 | **11.22%** | **−80.12** |

| Vách đá | Giảm > 1 điểm từ | Giảm > 5 điểm từ |
|---|---:|---:|
| Per-tensor | 4 bit | 3 bit |
| Per-channel | 3 bit | 2 bit |

PTQ so với QAT:

| Bit | PTQ per-tensor | PTQ per-channel | QAT per-channel | Δ QAT vs FP32 |
|---:|---:|---:|---:|---:|
| 4 | 88.85% | 90.41% | **91.45%** | **+0.11** |
| 3 | 64.53% | 86.82% | 91.32% | −0.02 |
| 2 | 10.00% | 11.22% | 89.40% | −1.94 |

| Chỉ tiêu | Kết quả | |
|---|---|---|
| QAT INT4 mất ≤ 1 điểm so với FP32 | +0.11 điểm (không mất) | **PASS** |

## Nhận xét

**Vách đá rơi đúng ở chỗ lưới quá thưa so với độ chênh giữa các filter.** Từ 8 xuống 5 bit gần như không mất
gì. Với per-tensor, một scale cho cả lớp bị filter có weight lớn nhất quyết định, nên filter nhỏ chỉ dùng được vài
mức; vì vậy per-tensor gãy sớm hơn per-channel một bit. Khoảng cách lớn nhất ở 3 bit: 86.82% so với 64.53%
(22.29 điểm). Ở 2 bit, lưới symmetric chỉ có 4 mức `{-2, -1, 0, 1} × max|w|`, mọi weight nhỏ hơn nửa max bị làm
tròn về 0, nên cả hai cách đều về mức đoán ngẫu nhiên.

**QAT xoá vách đá ở 4 và 3 bit.** PTQ INT4 per-channel thực ra cũng vừa lọt ngưỡng (−0.93) nhưng chỉ dư 0.07 điểm.
QAT kéo lên 91.45% ngay từ epoch 1 (91.19%). INT3 từ 86.82% lên 91.32%. INT2 hồi phục 78 điểm lên 89.40%, vẫn đang
tăng ở epoch cuối (86.86 → 89.40), nên thêm epoch nhiều khả năng sẽ còn lên. Accuracy của model INT-n xuất ra khớp
accuracy fake-quant lúc train (sai khác ≤ 0.01 điểm).

**Phải đọc +0.11 của QAT INT4 cẩn thận.** QAT là thêm 8 epoch fine-tune bằng KD, nên một phần mức tăng có thể đến
từ việc train thêm chứ không phải từ quantization. So sánh hoàn toàn công bằng cần fine-tune FP32 thêm 8 epoch làm
mốc. Ngoài ra +0.11 nằm trong nhiễu của một seed.

## Hạn chế và hướng tiếp theo

- **Weight-only, chưa fold BN.** Activation vẫn FP32. Fold BN vào conv sẽ đổi phân bố weight và có thể dời vách đá.
- **Size chưa giảm thật ở 4/3/2 bit:** bản packed vẫn lưu trong `int8`. Cần nén nhiều số vào một byte. Về lý thuyết
  weight INT4 nhỏ hơn FP32 8 lần.
- **INT2:** thử QAT lâu hơn, giữ conv đầu và lớp cuối ở 8 bit, hoặc học scale (LSQ) thay vì dùng `max|w|`.
- **Một seed.** Student FP32 lần này đạt 91.34% (bài 1.1: 91.38%) dù teacher giống hệt, do GPU không hoàn toàn tất định.
- **Tải CIFAR-10 mất 26 phút** (khoảng 108 kB/s), lâu hơn cả phần tính toán (khoảng 23 phút).
