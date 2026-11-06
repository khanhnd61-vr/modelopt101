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
