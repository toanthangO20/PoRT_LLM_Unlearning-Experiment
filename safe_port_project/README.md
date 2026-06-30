# SAFE-PoRT local project

Thư mục này là project code độc lập cho phương pháp đề xuất **SAFE-PoRT**:

```text
LoRA adapter unlearning
+ adversarial variants
+ belief-negative mining
+ post-judgment guard
+ robust evaluation
```

Mục tiêu của project là có một skeleton đủ đầy đủ để chạy local/Kaggle khi có GPU phù hợp, nhưng không phụ thuộc trực tiếp vào notebook cũ của PoRT.

## Cấu trúc

```text
safe_port_project/
  configs/
    local_tiny_debug.json          # config nhỏ để debug pipeline
    wmdp_safe_port_template.json   # config WMDP đầy đủ hơn
  examples/
    retain_safe_toy.jsonl
    neighbor_safe_toy.jsonl
  scripts/
    run_pipeline_example.ps1
  src/safe_port/
    augment.py
    belief.py
    cli.py
    config.py
    data.py
    evaluation.py
    io_utils.py
    metrics.py
    router.py
    train_adapter.py
  requirements.txt
  pyproject.toml
```

## Cài đặt

Tạo môi trường Python riêng:

```powershell
cd E:\PoRT_LLM_Unlearning-Experiment\safe_port_project
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Nếu chạy model lớn, cần GPU và cấu hình CUDA/PyTorch phù hợp. File `requirements.txt` không ép `bitsandbytes` vì package này thường khó dùng trên Windows.

## Các bước pipeline

Không chạy các lệnh dưới đây nếu máy không đủ GPU. Đây là lệnh mẫu để dùng khi chuyển sang Kaggle/Colab/server.

### 1. Build dữ liệu

```powershell
safe-port build-data --config configs/wmdp_safe_port_template.json
```

Output mặc định:

```text
runs/safe_port_wmdp/artifacts/data/safe_port_records.jsonl
```

### 2. Mine belief negatives

```powershell
safe-port mine-beliefs --config configs/wmdp_safe_port_template.json
```

Output mặc định:

```text
runs/safe_port_wmdp/artifacts/beliefs/belief_negatives.jsonl
```

### 3. Train LoRA adapter

```powershell
safe-port train-adapter --config configs/wmdp_safe_port_template.json
```

Output mặc định:

```text
runs/safe_port_wmdp/adapter/
```

### 4. Evaluate

```powershell
safe-port evaluate --config configs/wmdp_safe_port_template.json
```

Output mặc định:

```text
runs/safe_port_wmdp/eval/
```

## Chạy debug nhỏ

`configs/local_tiny_debug.json` dùng ít sample hơn và model nhỏ hơn để kiểm tra luồng. Vẫn có thể cần Internet để tải model từ Hugging Face.

```powershell
safe-port build-data --config configs/local_tiny_debug.json
safe-port mine-beliefs --config configs/local_tiny_debug.json --dry-run
safe-port evaluate --config configs/local_tiny_debug.json --dry-run
```

## Metric chính

- `forget_success`: tỷ lệ câu hỏi nguy hiểm không còn được trả lời đúng/đầy đủ.
- `adversarial_robustness`: mức giảm leakage trên prefix/paraphrase/composite/relearning.
- `open_ended_safety`: tỷ lệ generation không leak nội dung nhạy cảm.
- `utility_retention`: năng lực giữ lại trên retain set.
- `neighbor_utility`: năng lực giữ lại trên tri thức an toàn gần miền forget.

## Lưu ý an toàn

Code không in trực tiếp nội dung câu hỏi WMDP ra terminal trừ khi người dùng tự mở artifact. Các failure cases trong báo cáo nên được rút gọn/ẩn nội dung thao tác nhạy cảm.

