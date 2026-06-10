# PoRT LLM Unlearning Experiment Plan

## Mục tiêu

Mục tiêu hiện tại là reproduce lại kết quả paper gốc theo từng nấc kiểm chứng được trên Kaggle GPU. Paper-baseline smoke test đã pass; bước trước mắt là chạy full baseline/no-defense trên `original + noise_prefix + composite`, sau đó mới tạo smoke test cho pipeline PoRT paper-faithful.

Definition of done cho bước "full pipeline":

- Chạy được từ một Kaggle session sạch bằng cách clone GitHub repo.
- Load được target model bằng config trong repo hoặc runtime config sinh trong notebook/script.
- Load đủ WMDP full dataset: `wmdp-bio`, `wmdp-chem`, `wmdp-cyber`.
- Smoke test được paper baseline/no-defense trên WMDP `original`, `noise_prefix`, và `composite` trước khi chạy full.
- Chạy được ít nhất một baseline no-defense và một PoRT method trên full WMDP.
- Nếu dùng classifier-gated PoRT, `WMDP_CLASSIFIER_PATH` được cấu hình rõ, classifier load được, có log số prompt bị attack.
- Ghi đủ artifacts: `run_config.json`, `summary.json`, predictions CSV, partial artifacts nếu run dài.
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

## Kết quả theo notebook hiện tại

| Notebook | Mục đích | Trạng thái | Kết quả chính |
| --- | --- | --- | --- |
| `notebooks/smoke_tests/01_kaggle_smoke_test.ipynb` | Smoke test tổng hợp cho import, placeholder, TOFU, WMDP, tiny real model | Đã pass | `SMOKE TEST COMPLETED`; không còn syntax/import/placeholder blocker; TOFU và WMDP sample path chạy được |
| `notebooks/smoke_tests/02_kaggle_wmdp_full_tiny_gpt2.ipynb` | Chạy full WMDP với `sshleifer/tiny-gpt2`, no-corrupt, để test scale dataset/output | Đã pass | Full `3668` rows; overall acc `0.255998`; bio `0.250589`, chem `0.247549`, cyber `0.261198`; chạy trên CPU Kaggle |
| `notebooks/smoke_tests/03_kaggle_wmdp_target_model_mini_gpu.ipynb` | Mini test target model `microsoft/phi-1_5`, WMDP `sample_size=2`, no-corrupt | Đã pass | T4 x2; model `1.418B` params; total `6` rows; overall acc `0.166667` |
| `notebooks/smoke_tests/04_kaggle_wmdp_target_model_full_gpu.ipynb` | Full WMDP baseline với target model, no-corrupt | Đã pass | T4 x2; full `3668` rows; overall acc `0.394766`; bio `0.523959`, chem `0.335784`, cyber `0.324107` |
| `notebooks/smoke_tests/05_kaggle_wmdp_target_model_corrupt_hook_mini_gpu.ipynb` | Mini test `AttackedModel` corruption hook không dùng classifier, WMDP `sample_size=2` | Đã pass | Hook path chạy end-to-end; baseline/corrupt đều hoàn tất; total `18` prediction rows |
| `notebooks/smoke_tests/06_kaggle_wmdp_target_model_corrupt_hook_full_gpu.ipynb` | Full WMDP corruption hook không dùng classifier | Đã pass | T4 x2; full `3668` rows mỗi run; baseline overall `0.394766`; `zero_out_first_n` overall `0.246183` với bio `0.245090`, chem `0.235294`, cyber `0.249119`; `flip_sign_first_n` overall `0.241821` với bio `0.239592`, chem `0.267157`, cyber `0.238047` |
| `notebooks/smoke_tests/08_kaggle_wmdp_classifier_gated_mini_gpu.ipynb` | Mini test classifier-gated PoRT qua script canonical, WMDP `sample_size=2`, `zero_out_first_n` | Đã chuẩn bị, chờ artifact classifier | Notebook tự resolve/validate `WMDP_CLASSIFIER_PATH`, load thử `PromptClassifier`, chạy script không dùng `--attack_all_prompts`, rồi assert `attack_stats.csv` |
| `notebooks/smoke_tests/09_kaggle_wmdp_classifier_gated_multi_config_mini_gpu.ipynb` | Mini test classifier-gated PoRT với nhiều corrupt configs | Đã tạo, chờ Kaggle pass/fail | Reuse classifier flow từ notebook `08`, chạy tuần tự `zero_out`, `flip_sign`, `rand_noise_1`, `rand_noise_full`, rồi ghi aggregate summary/attack stats |
| `notebooks/smoke_tests/10_kaggle_paper_baseline_wmdp_smoke_test.ipynb` | Paper baseline/no-defense WMDP smoke test vài sample trên `original`, `noise_prefix`, `composite` | Đã sửa, cần rerun | Adversarial variants giờ dùng `full_question`; output cũ đã clear vì lần chạy trước dùng nhầm `question` cho mọi variant |
| `notebooks/paper_baselines/11_kaggle_paper_baseline_wmdp_full_no_defense.ipynb` | Full paper baseline/no-defense trên `original`, `noise_prefix`, `composite` | Đã sửa, cần rerun Kaggle | Mặc định full dataset, expected `11004` rows; adversarial variants giờ dùng `full_question`; có partial artifacts sau từng `variant/domain` |

## Phân tích trạng thái hiện tại

Baseline target-model full WMDP no-corrupt đã đủ tin cậy để làm mốc so sánh nội bộ. Lần chạy notebook `11` trước đó pass về row count nhưng không paper-faithful cho `noise_prefix` và `composite`, vì WMDP eval đang dùng cột `question` thay vì prompt adversarial `full_question`. Code dataset và notebook `10`/`11` đã được sửa để adversarial variants dùng `full_question` như prompt hoàn chỉnh. Các notebook corruption hook và classifier-gated trước đó vẫn là nhánh engineering/adapted, chưa phải reproduction paper gốc.

Điểm chưa hoàn tất:

- Chưa rerun notebook `10`/`11` trên Kaggle sau khi sửa `full_question`.
- Lần chạy full notebook `11` cũ chỉ xác nhận `original` baseline; kết quả `noise_prefix`/`composite` cũ không dùng được.
- Chưa smoke test pipeline PoRT paper-faithful; classifier-gated/corrupt-hook 08/09 chỉ nên xem là thử nghiệm phụ cho tới khi paper baseline full pass.

## Kế hoạch triển khai tiếp theo

### Bước 1: Chạy full paper baseline/no-defense

Trạng thái: **Đã sửa prompt source, chờ rerun Kaggle**.

Notebook:

`notebooks/paper_baselines/11_kaggle_paper_baseline_wmdp_full_no_defense.ipynb`

Mục tiêu:

- Chạy full WMDP baseline/control không PoRT, không classifier, không corruption hook.
- Bao phủ `original`, `noise_prefix`, `composite`.
- Bao phủ `bio`, `chem`, `cyber`.
- Ghi partial artifacts sau từng `variant/domain` để có thể inspect nếu Kaggle session bị ngắt.

Tiêu chí pass:

- Không có runtime error.
- Tổng row mặc định là `11004`.
- Mỗi variant có `3668` rows.
- Log build dataset cho `original` phải là `question_key=question formatted=False`.
- Log build dataset cho `noise_prefix` và `composite` phải là `question_key=full_question formatted=True`.
- Có đủ `predictions.csv`, `summary_by_variant_domain.csv`, `summary_by_variant.csv`, `summary_overall.csv`, `completed_jobs.csv`, `summary.json`, `run_config.json`.
- Kết quả `original` khớp gần baseline full đã có: overall khoảng `0.394766`, bio `0.523959`, chem `0.335784`, cyber `0.324107`.

### Bước 2: Tạo PoRT paper-faithful smoke test

Trạng thái: **Chưa làm**.

Notebook dự kiến:

`notebooks/smoke_tests/12_kaggle_paper_port_pipeline_smoke_test.ipynb`

Mục tiêu:

- Smoke test pipeline PoRT gốc với vài sample trước khi chạy full.
- Không dùng nhánh classifier-gated/corrupt-hook 08/09 làm thay thế cho paper pipeline.
- Chạy qua entrypoint hoặc wrapper dựa trên `PoRT_pipeline/WMDP/port_pipeline_wmdp.py`.

Việc cần xử lý trước:

- Loại bỏ hoặc truyền đầy đủ các `PATH_PLACEHOLDER` trong `PoRT_pipeline/WMDP/port_pipeline_wmdp.py`.
- Tải hoặc clone artifact cần thiết bằng code, không phụ thuộc thêm file trong `/kaggle/input`:
  - T5 AST/prefix model path.
  - Target model path hoặc Hugging Face hub name.
  - Classifier base model.
  - Classifier head checkpoint.
  - Example library JSON.
- Dùng dataset path trong repo cho `original`, `noise_prefix`, `composite`.
- Bật `--max_samples 2` hoặc tương đương để smoke test nhanh.

Tiêu chí pass:

- Chạy được ít nhất một domain với vài sample end-to-end.
- Có output generations, predicted choice, accuracy, rethink stats, timing stats.
- Không còn hardcoded local path hoặc placeholder.
- Artifact recipe đủ rõ để rerun từ Kaggle session sạch.

### Bước 3: Chạy PoRT paper smoke đủ domain/variant

Trạng thái: **Chờ Bước 2 pass**.

Mục tiêu:

- Mở rộng smoke test PoRT từ một domain sang `bio`, `chem`, `cyber`.
- Chạy trên các variant cần cho bảng paper, tối thiểu `composite` và baseline input tương ứng.
- Giữ `max_samples` nhỏ để xác nhận parity logic trước khi trả giá full runtime.

Tiêu chí pass:

- Mỗi domain/variant có row count đúng với `max_samples`.
- Accuracy/rethink stats được ghi theo domain/variant.
- Không có lỗi parse đáp án A/B/C/D.
- Runtime đủ thực tế để ước lượng full run.

### Bước 4: Chạy PoRT paper full dataset

Trạng thái: **Chờ Bước 3 pass**.

Mục tiêu:

- Chạy full PoRT paper pipeline trên WMDP theo recipe đã khóa.
- So sánh trực tiếp với full no-defense baseline từ notebook `11`.

Tiêu chí pass:

- Full row count đúng cho từng domain/variant.
- Có `final_generations_full.json`, metrics JSON/CSV, rethink stats, timing stats.
- Có bảng so sánh baseline vs PoRT theo variant/domain.
- Không chạy full nếu smoke còn placeholder, artifact không tái lập được, hoặc output parsing chưa ổn.

### Bước 5: Thêm utility/general eval nếu paper table yêu cầu

Trạng thái: **Chờ WMDP PoRT full ổn định**.

Mục tiêu:

- Đánh giá tradeoff giữa forgetting/robustness và utility.
- Thêm MMLU hoặc subset utility tương ứng với paper sau khi WMDP pipeline đã ổn.

Tiêu chí pass:

- Utility metric được ghi cùng recipe và commit.
- Có thể so sánh với baseline/no-defense và PoRT full.

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

## Next immediate action

Rerun paper baseline/no-defense sau fix `full_question`:

- Chạy nhanh `notebooks/smoke_tests/10_kaggle_paper_baseline_wmdp_smoke_test.ipynb` trước để xác nhận `noise_prefix`/`composite` không còn cho kết quả y hệt `original`.
- Sau đó chạy `notebooks/paper_baselines/11_kaggle_paper_baseline_wmdp_full_no_defense.ipynb` từ một Kaggle session sạch.
- Giữ mặc định `PORT_WMDP_SAMPLE_SIZE` unset để chạy full.
- Giữ mặc định `PORT_WMDP_BASELINE_VARIANTS=original,noise_prefix,composite` và `PORT_WMDP_DOMAINS=bio,chem,cyber`.
- Xác nhận tổng rows là `11004`.
- Xác nhận log prompt source: `original` dùng `question`, `noise_prefix` và `composite` dùng `full_question`.
- Nếu notebook `11` pass, next action là tạo `12_kaggle_paper_port_pipeline_smoke_test.ipynb`.
- Nếu notebook `11` fail, sửa data path/model config/evaluator trước khi đụng đến PoRT full.
