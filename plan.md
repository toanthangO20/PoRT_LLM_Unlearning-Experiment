# PoRT LLM Unlearning Experiment Plan

## Mục tiêu

Mục tiêu hiện tại là đưa repo đến trạng thái có thể chạy được pipeline full dataset trên Kaggle GPU, bắt đầu từ baseline WMDP no-corrupt, sau đó mở rộng sang corruption hook, rồi đến classifier-gated PoRT pipeline giống hướng chạy full experiment.

Definition of done cho bước "full pipeline":

- Chạy được từ một Kaggle session sạch bằng cách clone GitHub repo.
- Load được target model bằng config trong repo hoặc runtime config sinh trong notebook/script.
- Load đủ WMDP full dataset: `wmdp-bio`, `wmdp-chem`, `wmdp-cyber`.
- Chạy được ít nhất một baseline no-corrupt và một corrupt method trên full WMDP.
- Nếu dùng classifier-gated PoRT, `WMDP_CLASSIFIER_PATH` được cấu hình rõ, classifier load được, có log số prompt bị attack.
- Ghi đủ artifacts: `run_config.json`, `summary.json`, predictions CSV, partial artifacts nếu run dài.
- Tổng số row WMDP full cho mỗi run là `3668`.

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

## Phân tích trạng thái hiện tại

Baseline target-model full WMDP no-corrupt đã đủ tin cậy để làm mốc so sánh. Corrupt-hook full không classifier đã xác nhận nhánh `AttackedModel` scale được lên full WMDP và kéo accuracy về gần random-choice baseline.

Điểm chưa hoàn tất:

- Chưa chạy classifier-gated PoRT vì cần artifact classifier thật và biến môi trường `WMDP_CLASSIFIER_PATH`.
- Notebook/script-level cho classifier-gated mini đã sẵn sàng, nhưng chưa có kết quả Kaggle pass/fail.

## Kế hoạch triển khai tiếp theo

### Bước 1: Chạy full corrupt-hook không classifier

Trạng thái: **Đã pass**.

Notebook:

`notebooks/smoke_tests/06_kaggle_wmdp_target_model_corrupt_hook_full_gpu.ipynb`

Mục tiêu:

- Chạy full WMDP với baseline tùy chọn và corrupt methods:
  - `zero_out_first_n`
  - `flip_sign_first_n`
- Kiểm tra scale của `AttackedModel` trên full dataset.
- So sánh effect corrupt với baseline `phi-1_5` no-corrupt đã có.

Khuyến nghị Kaggle:

- GPU: T4 x2.
- Giữ `PORT_WMDP_BATCH_SIZE=1`.
- Nếu muốn tiết kiệm thời gian vì baseline full đã có, đặt `PORT_RUN_BASELINE=0`.

Tiêu chí pass:

- Không có runtime error.
- Mỗi corrupt method chạy đủ `3668` rows.
- Có `summary.json`, `summary_by_run.csv`, `predictions.csv`, `predictions_partial.csv`.
- Accuracy của corrupt methods có khác biệt đủ rõ để xác nhận hook ảnh hưởng trên full set.

Kết quả:

- `baseline_none`: overall `0.394766`; bio `0.523959`, chem `0.335784`, cyber `0.324107`.
- `zero_out_first_n`: overall `0.246183`; bio `0.245090`, chem `0.235294`, cyber `0.249119`.
- `flip_sign_first_n`: overall `0.241821`; bio `0.239592`, chem `0.267157`, cyber `0.238047`.
- Runtime mỗi full pass khoảng `5.1`-`5.2` phút trên Kaggle T4 x2 với `PORT_WMDP_BATCH_SIZE=1`.

### Bước 2: Chuẩn hóa script pipeline `evaluate_wmdp.py`

Trạng thái: **Đã pass Kaggle mini với target model**.

Mục tiêu:

- Biến `llm-unlearn-eco/scripts/evaluate_wmdp.py` thành pipeline canonical có thể chạy full experiment thay vì chỉ notebook ad hoc.

Việc cần làm:

- Gỡ hoặc thay `patch_hf_model()` cũ để dùng trực tiếp `eco.model.HFModel` đã sửa.
- Thêm các runtime override giống notebook:
  - `torch_dtype`
  - `attn_implementation`
  - `model_path`
  - `batch_size`
  - `sample_size`
- Thêm mode corruption không classifier, ví dụ `--attack_all_prompts`, để script chạy được corrupt-hook tests mà không cần `WMDP_CLASSIFIER_PATH`.
- Chuẩn hóa output:
  - `run_config.json`
  - `summary.json`
  - predictions CSV
  - partial outputs sau từng dataset/task.
- Giữ lại `--save_logits` nhưng chỉ bật khi cần phân tích sâu vì file có thể lớn.

Tiêu chí pass:

- Script chạy được original no-corrupt với `--sample_size 2`.
- Script chạy được corrupt-hook với `--sample_size 2 --attack_all_prompts`.
- Kết quả tương đương notebook mini theo row count và không lỗi.

Kết quả local hiện tại:

- `evaluate_wmdp.py` đã bỏ monkey patch `HFModel` cũ và dùng runtime config sinh trong `results/<run_name>/model_config`.
- Đã thêm `--attack_all_prompts`, `--torch_dtype`, `--attn_implementation`, `--model_path`, `--target_hf_name`, `--batch_size`, `--sample_size`, `--output_dir`, `--run_name`.
- Output chuẩn gồm `run_config.json`, `summary.json`, `summary_by_run.csv`, `summary_overall.csv`, `predictions.csv`, partial artifacts, và `attack_stats.csv` cho corrupt runs.
- Local smoke bằng `tiny-gpt2`, `sample_size=1` đã pass cho original và `zero_out_first_n --attack_all_prompts`, mỗi run ghi đủ `3` prediction rows.

Kết quả Kaggle target-model mini:

- Notebook `notebooks/smoke_tests/07_kaggle_wmdp_pipeline_script_mini_gpu.ipynb` đã pass trên commit `f0196e944244c067612e64f818bcd6e9dff50964`.
- Original no-corrupt script run: `6` rows; overall acc `0.166667`; bio `0.5`, chem `0.0`, cyber `0.0`.
- Corrupt-hook `zero_out_first_n --attack_all_prompts`: `6` rows; overall acc `0.333333`; bio `0.5`, chem `0.0`, cyber `0.5`.
- Cả hai run đều ghi `run_config.json`, `summary.json`, `summary_by_run.csv`, `predictions.csv`.

### Bước 3: Tạo notebook/script-level smoke test cho pipeline canonical

Trạng thái: **Đã pass trên Kaggle**.

Notebook đề xuất:

`notebooks/smoke_tests/07_kaggle_wmdp_pipeline_script_mini_gpu.ipynb`

Mục tiêu:

- Không gọi trực tiếp Python classes trong notebook nữa.
- Gọi `python llm-unlearn-eco/scripts/evaluate_wmdp.py ...` để test đúng entrypoint pipeline.

Các run cần có:

- No-corrupt:
  - `task_config=multiple_choice_original.yaml`
  - `sample_size=2`
- Corrupt-hook không classifier:
  - `task_config=multiple_choice_zero_out.yaml`
  - `sample_size=2`
  - `attack_all_prompts=true`

Tiêu chí pass:

- Script entrypoint chạy từ Kaggle clean clone.
- Output được ghi đúng nơi.
- Không cần classifier artifact ở bước này.

### Bước 4: Chuẩn bị classifier-gated PoRT

Trạng thái: **Đã chuẩn bị notebook/script-level, chờ artifact classifier để chạy Kaggle**.

Mục tiêu:

- Chạy được pipeline đúng hướng PoRT: chỉ attack prompt được classifier đánh dấu là WMDP-relevant.

Yêu cầu artifact:

- Có local classifier path trên Kaggle.
- Set biến môi trường:
  - `WMDP_CLASSIFIER_PATH=/kaggle/input/...`

Việc cần kiểm tra:

- Classifier load được bằng `PromptClassifier`.
- Classifier chạy trên sample WMDP và non-WMDP.
- Log số prompt bị attack theo dataset:
  - `num_prompts`
  - `num_attacked`
  - `attack_rate`
- Nếu attack rate toàn `0` hoặc toàn `1`, cần debug threshold hoặc classifier labels trước khi chạy full.

Tiêu chí pass:

- Classifier-gated run với `sample_size=2` hoàn tất.
- Có log attack labels.
- Có predictions và summary.

Việc đã làm:

- Thêm notebook `notebooks/smoke_tests/08_kaggle_wmdp_classifier_gated_mini_gpu.ipynb`.
- Notebook validate sớm classifier path là local Hugging Face text-classification model directory.
- Notebook load thử `PromptClassifier` trên một prompt WMDP và một prompt non-WMDP trước khi tải target model.
- Notebook chạy `evaluate_wmdp.py` với `multiple_choice_zero_out.yaml`, `sample_size=2`, không bật `--attack_all_prompts`.
- Nếu chưa mount artifact classifier thật, notebook có thể tải classifier từ Hugging Face repo/archive URL hoặc auto-train một classifier WMDP-vs-non-WMDP thật từ WMDP/TOFU trong repo và lưu vào `/kaggle/working`.
- `evaluate_wmdp.py` đã validate `WMDP_CLASSIFIER_PATH`/`--classifier_path` trước khi load target model nếu task config có corrupt method và không bật `--attack_all_prompts`.

### Bước 5: Chạy classifier-gated mini với nhiều corrupt configs

Trạng thái: **Đã tạo notebook, chờ chạy Kaggle**.

Configs hiện có:

- `multiple_choice_zero_out.yaml`
- `multiple_choice_flip_sign.yaml`
- `multiple_choice_rand_noise_1.yaml`
- `multiple_choice_rand_noise_full.yaml`

Thứ tự khuyến nghị:

1. `zero_out_first_n`
2. `flip_sign_first_n`
3. `rand_noise_first_n` với strength nhỏ hoặc vừa
4. `rand_noise_first_n` full strength sau khi đã có safety evidence

Tiêu chí pass:

- Mỗi config chạy được `sample_size=2`.
- Không OOM.
- Không NaN logits.
- Có output artifact cho từng config.

Notebook:

`notebooks/smoke_tests/09_kaggle_wmdp_classifier_gated_multi_config_mini_gpu.ipynb`

Output aggregate:

- `/kaggle/working/wmdp_classifier_gated_multi_config_mini_summary.csv`
- `/kaggle/working/wmdp_classifier_gated_multi_config_mini_attack_stats.csv`

### Bước 6: Chạy classifier-gated full WMDP

Mục tiêu:

- Chạy full WMDP với classifier-gated corruption.

Khuyến nghị:

- Bắt đầu với một method ổn định nhất từ mini, ưu tiên `zero_out_first_n` hoặc `flip_sign_first_n`.
- `PORT_WMDP_BATCH_SIZE=1`.
- Chạy WMDP-only trước, chưa thêm MMLU.
- Bật partial artifact.

Tiêu chí pass:

- Full `3668` rows cho mỗi task.
- Có summary theo dataset.
- Có attack-rate log.
- Có thể so sánh với no-corrupt baseline:
  - baseline overall `0.394766`
  - bio `0.523959`
  - chem `0.335784`
  - cyber `0.324107`

### Bước 7: Thêm utility eval sau khi WMDP pass

Mục tiêu:

- Đánh giá tradeoff giữa unlearning và general utility.

Thứ tự:

1. WMDP-only full pass.
2. Thêm MMLU nếu dataset và runtime ổn.
3. Nếu MMLU quá chậm, dùng MMLU subset hoặc sample trước.

Tiêu chí pass:

- WMDP corruption không làm pipeline vỡ.
- Utility metric được ghi cùng artifact để so sánh.

### Bước 8: Tổng hợp kết quả và khóa experiment recipe

Artifacts cần chuẩn hóa:

- `results/<run_name>/run_config.json`
- `results/<run_name>/summary.json`
- `results/<run_name>/summary_by_dataset.csv`
- `results/<run_name>/predictions.csv`
- `results/<run_name>/attack_stats.csv` nếu có classifier.

Tài liệu cần tạo sau full runs:

- `results/README.md` hoặc `notebooks/results_summary.md`
- Bảng so sánh:
  - model
  - dataset
  - corrupt method
  - dims
  - strength
  - classifier threshold
  - attack rate
  - accuracy
  - runtime

## Next immediate action

Chạy classifier-gated PoRT mini run trên Kaggle:

- Cung cấp hoặc mount classifier artifact trên Kaggle.
- Set `WMDP_CLASSIFIER_PATH=/kaggle/input/...` hoặc điền `MANUAL_CLASSIFIER_PATH` trong notebook `08` nếu muốn chạy classifier thật.
- Chạy `notebooks/smoke_tests/08_kaggle_wmdp_classifier_gated_mini_gpu.ipynb`.
- Xác nhận `attack_stats.csv` có `num_prompts`, `num_attacked`, `attack_rate`, và `classifier_mode=classifier_gated`.
- Nếu `attack_rate` toàn `0` hoặc toàn `1`, debug threshold hoặc label mapping trước khi chạy full.
- Nếu notebook dùng classifier auto-trained trong session, xem đó là pass cho pipeline classifier-gated; trước full experiment nên khóa nguồn classifier bằng `PORT_WMDP_CLASSIFIER_HF_REPO` hoặc `PORT_WMDP_CLASSIFIER_ARCHIVE_URL` để recipe reproducible hơn.
