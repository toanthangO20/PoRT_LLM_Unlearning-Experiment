# PoRT LLM Unlearning Experiment Plan

## Mục tiêu

Mục tiêu cuối cùng là reproduce lại kết quả paper gốc theo từng nấc kiểm chứng được trên Kaggle GPU. Full paper baseline/no-defense đã chạy xong trên `original + noise_prefix + composite`; bước hiện tại là smoke test pipeline PoRT paper-faithful với vài sample trước khi chạy full PoRT.

Definition of done cho full reproduction:

- Chạy được từ một Kaggle session sạch bằng cách clone GitHub repo.
- Load được target model bằng config trong repo hoặc runtime config sinh trong notebook/script.
- Load đủ WMDP full dataset: `wmdp-bio`, `wmdp-chem`, `wmdp-cyber`.
- Chạy được baseline/no-defense trên WMDP `original`, `noise_prefix`, và `composite`.
- Chạy được PoRT method paper-faithful trên full WMDP.
- Ghi đủ artifacts: `run_config.json`, `summary.json`, predictions/generations, metrics, timing stats, rethink stats nếu chạy PoRT.
- Tổng số row WMDP full là `3668` cho mỗi variant; với `original + noise_prefix + composite` tổng baseline/no-defense mặc định là `11004` rows.

## Kết quả đã đạt được

### Repo và code nền

- Đã chuyển repo sang layout có thể clone trực tiếp trên Kaggle.
- Đã bổ sung path config qua `eco.paths` để tránh placeholder/local-only path.
- Đã khôi phục WMDP dataset module, đọc local parquet trong `dataset/WMDP`.
- Đã sửa TOFU classification loader cho phiên bản `datasets` mới.
- Đã sửa `HFModel` để hỗ trợ runtime config gồm `torch_dtype`, `attn_implementation`, `trust_remote_code`.
- Đã sửa lỗi `PhiConfig` thiếu `pad_token_id` khi load `microsoft/phi-1_5`.
- Đã sửa evaluator `ChoiceByTopLogit` để truncate prompt theo context window, tránh lỗi prompt dài ở WMDP cyber.
- Đã sửa WMDP adversarial eval để `noise_prefix` và `composite` dùng `full_question` thay vì `question`.

### Notebook đã chạy

| Notebook | Mục đích | Trạng thái | Kết quả chính |
| --- | --- | --- | --- |
| `notebooks/smoke_tests/01_kaggle_smoke_test.ipynb` | Smoke test tổng hợp cho import, placeholder, TOFU, WMDP, tiny real model | Đã pass | `SMOKE TEST COMPLETED`; không còn syntax/import/placeholder blocker |
| `notebooks/smoke_tests/02_kaggle_wmdp_full_tiny_gpt2.ipynb` | Full WMDP với `sshleifer/tiny-gpt2`, no-corrupt | Đã pass | `3668` rows; overall acc `0.255998` |
| `notebooks/smoke_tests/03_kaggle_wmdp_target_model_mini_gpu.ipynb` | Mini test target model `microsoft/phi-1_5` | Đã pass | `6` rows; overall acc `0.166667` |
| `notebooks/smoke_tests/04_kaggle_wmdp_target_model_full_gpu.ipynb` | Full WMDP baseline target model, no-corrupt | Đã pass | `3668` rows; overall acc `0.394766` |
| `notebooks/smoke_tests/05_kaggle_wmdp_target_model_corrupt_hook_mini_gpu.ipynb` | Mini test corruption hook không dùng classifier | Đã pass | Hook path chạy end-to-end; total `18` prediction rows |
| `notebooks/smoke_tests/06_kaggle_wmdp_target_model_corrupt_hook_full_gpu.ipynb` | Full WMDP corruption hook không dùng classifier | Đã pass | baseline `0.394766`; `zero_out_first_n` `0.246183`; `flip_sign_first_n` `0.241821` |
| `notebooks/smoke_tests/08_kaggle_wmdp_classifier_gated_mini_gpu.ipynb` | Mini classifier-gated PoRT qua script canonical | Đã chuẩn bị, nhưng không phải paper-faithful baseline | Cần classifier artifact nếu dùng lại nhánh adapted |
| `notebooks/smoke_tests/09_kaggle_wmdp_classifier_gated_multi_config_mini_gpu.ipynb` | Mini classifier-gated nhiều corrupt configs | Đã tạo/chạy thử như nhánh phụ | Không dùng làm reproduction paper gốc |
| `notebooks/smoke_tests/10_kaggle_paper_baseline_wmdp_smoke_test.ipynb` | Paper baseline/no-defense smoke trên `original`, `noise_prefix`, `composite` | Đã pass sau khi dùng `full_question` | Dùng để xác nhận prompt adversarial trước full baseline |
| `notebooks/paper_baselines/11_kaggle_paper_baseline_wmdp_full_no_defense.ipynb` | Full paper baseline/no-defense trên `original`, `noise_prefix`, `composite` | Đã pass trên Kaggle | `11004` rows; no errors; prompt source đúng cho cả 3 variants |
| `notebooks/smoke_tests/12_kaggle_paper_port_pipeline_smoke_test.ipynb` | PoRT paper pipeline smoke test vài sample | Đã tạo, chờ chạy Kaggle với artifact thật | Runtime patch paper script, build prompt đúng variant, ghi generations/metrics/timing/rethink |

### Kết quả notebook 11 mới nhất

Notebook `11` đã chạy full baseline/no-defense trên Kaggle ở commit `0e85d416...` với prompt source đúng:

- `original`: `question_key=question`, `formatted=False`
- `noise_prefix`: `question_key=full_question`, `formatted=True`
- `composite`: `question_key=full_question`, `formatted=True`

Kết quả:

| Variant | Rows | Accuracy |
| --- | ---: | ---: |
| `original` | 3668 | 0.394766 |
| `noise_prefix` | 3668 | 0.371320 |
| `composite` | 3668 | 0.286805 |
| overall | 11004 | 0.350963 |

Theo domain:

| Variant | Bio | Chem | Cyber |
| --- | ---: | ---: | ---: |
| `original` | 0.523959 | 0.335784 | 0.324107 |
| `noise_prefix` | 0.483111 | 0.323529 | 0.309512 |
| `composite` | 0.340141 | 0.240196 | 0.262204 |

Runtime Kaggle ghi nhận: model load khoảng `17.69s`, eval khoảng `26.2` phút.

## Phân tích trạng thái hiện tại

Full no-defense baseline hiện đã đủ tin cậy để làm mốc paper baseline trong repo này. Kết quả `noise_prefix` và `composite` không còn trùng `original`, và log xác nhận adversarial variants đã dùng `full_question`.

Các notebook corruption hook/classifier-gated trước đó là nhánh engineering/adapted. Chúng hữu ích để kiểm thử cơ chế can thiệp, nhưng chưa thay thế được pipeline PoRT paper-faithful. Vì vậy bước tiếp theo đúng là smoke test trực tiếp logic PoRT gốc trong `PoRT_pipeline/WMDP/port_pipeline_wmdp.py`.

## Kế hoạch triển khai tiếp theo

### Bước 1: Full paper baseline/no-defense

Trạng thái: **Hoàn tất**.

Notebook:

`notebooks/paper_baselines/11_kaggle_paper_baseline_wmdp_full_no_defense.ipynb`

Kết quả khóa:

- Tổng rows: `11004`.
- Mỗi variant: `3668` rows.
- `original` dùng `question`.
- `noise_prefix` và `composite` dùng `full_question`.
- Artifacts đã có trong notebook output local sau khi overwrite từ Kaggle.

### Bước 2: PoRT paper-faithful smoke test

Trạng thái: **Đã tạo notebook, chờ chạy Kaggle**.

Notebook:

`notebooks/smoke_tests/12_kaggle_paper_port_pipeline_smoke_test.ipynb`

Mục tiêu:

- Smoke test pipeline PoRT gốc với vài sample trước khi chạy full.
- Không dùng nhánh classifier-gated/corrupt-hook 08/09 làm thay thế cho paper pipeline.
- Reuse logic từ `PoRT_pipeline/WMDP/port_pipeline_wmdp.py`, nhưng runtime-patch các blocker rõ ràng:
  - `PATH_PLACEHOLDER` cho `POST_CLASSIFIER_DIR` và `ECO_DIR`.
  - `torch.bfloat16` thành dtype runtime, mặc định `float16` cho Kaggle T4.
  - bug key `models["llama_model"]` thành `models["main_llama_model"]`.
- Dùng dataset trong repo:
  - `original` dùng `question + choices`.
  - `noise_prefix` và `composite` dùng `full_question`.

Notebook yêu cầu artifact thật của PoRT. Có thể truyền qua env var hoặc URL download, không cần thêm file vào `/kaggle/input`:

- `PORT_T5_MODEL_PATH` hoặc `PORT_T5_MODEL_HF_REPO` hoặc `PORT_T5_MODEL_URL`
- `PORT_CLASSIFIER_BASE_MODEL`
- `PORT_CLASSIFIER_HEAD_CKPT` hoặc `PORT_CLASSIFIER_HEAD_URL`
- Optional: `PORT_TARGET_MODEL_PATH`, `PORT_TARGET_MODEL_HUB_NAME`
- Optional smoke config: `PORT_WMDP_VARIANT=composite`, `PORT_WMDP_DOMAIN=bio`, `PORT_MAX_SAMPLES=2`

Tiêu chí pass:

- Chạy được ít nhất một domain với vài sample end-to-end.
- Có output `final_generations_full.json`, `final_metrics_full.json`, `predictions.csv`, `rethink_stats.json`, `timing_stats.json`, `summary.json`, `run_config.json`.
- Không còn hardcoded local path hoặc placeholder trong notebook runtime.
- Nếu thiếu artifact, notebook fail sớm với danh sách env vars cần set.

### Bước 3: PoRT paper smoke đủ domain/variant

Trạng thái: **Chờ Bước 2 pass**.

Mục tiêu:

- Mở rộng smoke test PoRT từ một domain sang `bio`, `chem`, `cyber`.
- Chạy trên các variants cần cho bảng paper, tối thiểu `composite`, sau đó thêm `original`/`noise_prefix` nếu runtime cho phép.
- Giữ `max_samples` nhỏ để xác nhận logic trước full run.

Tiêu chí pass:

- Mỗi domain/variant có row count đúng với `max_samples`.
- Accuracy/rethink stats được ghi theo domain/variant.
- Không có lỗi parse đáp án A/B/C/D.
- Runtime đủ thực tế để ước lượng full run.

### Bước 4: PoRT paper full dataset

Trạng thái: **Chờ Bước 3 pass**.

Mục tiêu:

- Chạy full PoRT paper pipeline trên WMDP theo recipe đã khóa.
- So sánh trực tiếp với full no-defense baseline từ notebook `11`.

Tiêu chí pass:

- Full row count đúng cho từng domain/variant.
- Có generations/metrics/timing/rethink artifacts.
- Có bảng so sánh baseline vs PoRT theo variant/domain.
- Không chạy full nếu smoke còn placeholder, artifact không tái lập được, hoặc output parsing chưa ổn.

### Bước 5: Utility/general eval nếu paper table yêu cầu

Trạng thái: **Chờ WMDP PoRT full ổn định**.

Mục tiêu:

- Đánh giá tradeoff giữa forgetting/robustness và utility.
- Thêm MMLU hoặc subset utility tương ứng với paper sau khi WMDP pipeline đã ổn.

### Bước 6: Tổng hợp kết quả và khóa experiment recipe

Artifacts cần chuẩn hóa:

- `run_config.json`
- `summary.json`
- `predictions.csv` hoặc `final_generations_full.json`
- `summary_by_variant_domain.csv`
- `timing_stats.json`
- `rethink_stats.json` nếu chạy PoRT.

Tài liệu cần tạo sau full runs:

- `results/README.md` hoặc `notebooks/results_summary.md`.
- Bảng so sánh model, dataset variant, domain, method, accuracy, rethink count/rate, runtime, commit SHA.

## Next Immediate Action

Chạy notebook PoRT smoke mới trên Kaggle:

`notebooks/smoke_tests/12_kaggle_paper_port_pipeline_smoke_test.ipynb`

Thiết lập tối thiểu trước khi chạy:

- Set `PORT_T5_MODEL_PATH` hoặc `PORT_T5_MODEL_HF_REPO` hoặc `PORT_T5_MODEL_URL`.
- Set `PORT_CLASSIFIER_BASE_MODEL`.
- Set `PORT_CLASSIFIER_HEAD_CKPT` hoặc `PORT_CLASSIFIER_HEAD_URL`.
- Giữ mặc định `PORT_WMDP_VARIANT=composite`, `PORT_WMDP_DOMAIN=bio`, `PORT_MAX_SAMPLES=2` cho lần đầu.

Nếu notebook `12` pass, next action là tạo biến thể loop qua `bio/chem/cyber` và các variants cần thiết. Nếu notebook `12` fail, sửa artifact recipe hoặc runtime patch trong notebook trước khi mở rộng smoke/full.
