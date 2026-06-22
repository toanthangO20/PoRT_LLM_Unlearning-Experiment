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
| `notebooks/smoke_tests/12_kaggle_paper_port_pipeline_smoke_test.ipynb` | PoRT paper pipeline smoke test vài sample | Đã pass trên Kaggle ở smoke mode | `composite/bio`, `2` rows, prompt source `full_question`, rethink `2/2`, valid predictions `1.0`; không phải paper metric vì dùng smoke post-judge |
| `notebooks/smoke_tests/13_kaggle_paper_port_pipeline_smoke_matrix.ipynb` | PoRT smoke matrix đủ variant/domain | Đã pass trên Kaggle ở smoke mode | `9` jobs, `18` rows; prompt source đúng; rethink `18/18`; valid rate `1.0` ở 8/9 jobs, `composite/bio=0.5`; không phải paper metric |
| `notebooks/smoke_tests/15_kaggle_paper_port_official_artifact_probe.ipynb` | Probe official PoRT artifacts | Đã pass trên Kaggle | Không tìm thấy public T5/classifier checkpoint; env artifact chưa set; `PORT_ARTIFACT_MODE=official` chưa chạy được |
| `notebooks/artifact_bootstrap/16_kaggle_paper_port_recreated_artifacts_bootstrap.ipynb` | Bootstrap recreated PoRT artifacts | Đã pass trên Kaggle | Không phải smoke test; tạo được T5 recreated checkpoint/dataset và weak classifier dataset; classifier head vẫn unresolved |
| `notebooks/smoke_tests/17_kaggle_paper_port_recreated_artifact_smoke_matrix.ipynb` | PoRT recreated-artifact smoke matrix | Đã pass trên Kaggle | `9` jobs, `18` rows, valid rate `1.0`; classifier weak test acc `0.2155`; rethink `18/18`, nên chưa đủ để full run |
| `notebooks/smoke_tests/18_kaggle_paper_port_recreated_classifier_diagnostics.ipynb` | Recreated post-judge classifier diagnostics | Đã pass trên Kaggle | `9216` rows rebuilt; group split no leakage; best TF-IDF `answer_only` test acc `0.9286`, macro F1 `0.9074`; next là smoke matrix với answer expansion |
| `notebooks/smoke_tests/19_kaggle_paper_port_recreated_best_classifier_smoke_matrix.ipynb` | PoRT recreated best-classifier smoke matrix | Đã pass trên Kaggle | `9` jobs, `18` rows; valid rate `1.0`; rethink `10/18`; classifier test acc `0.9286`; vẫn là recreated smoke, không phải official paper metric |
| `notebooks/recreated_runs/20_kaggle_paper_port_recreated_scale_run.ipynb` | PoRT recreated best-classifier scale run | Đã pass trên Kaggle | Không phải smoke test; `288` rows (`32`/job), valid rate `0.9931`, rethink `0.6771`, overall acc `0.2222`; dùng classifier và answer expansion của notebook `19`; vẫn là recreated, không phải official metric |
| `notebooks/recreated_runs/21_kaggle_paper_port_recreated_ablation_diagnostics.ipynb` | PoRT recreated ablation diagnostics | Đã pass trên Kaggle | Không phải smoke test; `288` rows; raw direct acc `0.2917`, compiled initial `0.2361`, rethink-all `0.2188`; best threshold final chỉ `0.2188`, nên threshold sweep không cứu được notebook `20` |
| `notebooks/recreated_runs/22_kaggle_paper_port_generation_baseline_identity_ablation.ipynb` | Generation baseline + identity ablation | Đã pass trên Kaggle | Không phải smoke test; `288` rows; identity-prefix/no-rethink khớp raw generation tuyệt đối; compiled-prefix/no-rethink tụt `-0.0625`; không bootstrap/train recreated artifacts; không phải official paper metric |
| `notebooks/recreated_runs/23_kaggle_paper_port_prefix_compiler_source_diagnostic.ipynb` | Prefix compiler source diagnostic | Đã pass trên Kaggle với auto-download artifact | Không phải smoke test; `288` dataset rows, `864` prediction rows; chạy đủ `raw_direct`, `base_t5`, `recreated_artifact`; recreated artifact vẫn kém raw direct `-0.0243`, nên prefix compiler recreated chưa đủ tốt để full PoRT |
| `notebooks/recreated_runs/24_kaggle_paper_port_prefix_quality_gate_diagnostic.ipynb` | Prefix quality gate/prompt repair diagnostic | Đã pass trên Kaggle | Không phải smoke test; `288` dataset rows, `1152` prediction rows; quality gate sửa được format/valid rate nhưng chưa cứu accuracy; diagnostic còn bị confound vì fallback raw dùng generation seed khác raw direct |
| `notebooks/recreated_runs/25_kaggle_paper_port_prefix_quality_gate_counterfactual.ipynb` | Prefix quality gate counterfactual diagnostic | Đã pass trên Kaggle | Không phải smoke test; `288` dataset rows, `1152` prediction rows; `structure_gate` đạt `0.2986`, nhỉnh hơn raw `+0.0069`, nhưng reuse raw-direct ở `80.2%` rows nên đây là safety gate diagnostic, chưa phải full PoRT/paper metric |
| `notebooks/recreated_runs/26_kaggle_paper_port_recreated_structure_gate_scale_run.ipynb` | Recreated PoRT structure-gate scale path | Đã pass trên Kaggle | Không phải smoke test; `288` rows; overall acc `0.2222`, valid `0.9965`, rethink `0.7222`; structure gate pass `0.1632`, fallback raw `0.8368`; không cải thiện notebook `20` và thấp hơn raw direct notebook `21` |
| `notebooks/recreated_runs/27_kaggle_paper_port_postjudge_rethink_oracle_diagnostic.ipynb` | Post-judge/rethink oracle diagnostic | Đã pass trên Kaggle | Không phải smoke test; `288` rows; raw direct `0.2917`, raw selective `0.1840`, raw oracle `0.4271`, structure-gated selective `0.1944`, structure-gated oracle `0.4063`; oracle cao nhưng router/selective làm tụt mạnh |

### Kết quả notebook 26 mới nhất

Notebook `26` đã chạy xong trên Kaggle ở commit `a50323a95fdc4d0683aee90cefbb0d9c7c8dce4d`, không lỗi cell và không OOM:

- Matrix: `9` jobs x `32` rows = `288` dataset rows.
- Mode: `PORT_ARTIFACT_MODE=recreated`.
- Artifact: auto-download recreated artifact từ branch `artifact-recreated-bootstrap-v1`.
- Path thật đã test: T5 compile -> `structure_gate` -> best TF-IDF post-judge classifier -> rethink.
- Khác notebook `25`: fallback raw không reuse raw-direct prediction; fallback rows vẫn đi qua generate/post-judge/rethink như recreated PoRT thật.
- Classifier held-out test: accuracy `0.9286`, macro F1 `0.9074`.
- Artifacts đã ghi trên Kaggle: `artifact_audit.json`, `run_config.json`, `summary.json`, `matrix_summary.csv/json`, `summary_by_variant_domain.csv/json`, `all_predictions.csv`, `failed_jobs.json`.

Overall summary:

| Metric | Value |
| --- | ---: |
| Rows | `288` |
| Accuracy | `0.2222` |
| Valid predictions rate | `0.9965` |
| Rethink rate | `0.7222` |
| Post-judge positive rate | `0.3229` |
| Structure gate pass rate | `0.1632` |
| Structure gate fallback-to-raw rate | `0.8368` |
| T5 compiled choice coverage avg | `0.3082` |
| Gated prompt choice coverage avg | `1.0000` |

Theo variant/domain:

| Variant | Domain | Accuracy | Valid | Rethink | Gate pass | Fallback raw | T5 choice coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `original` | `bio` | `0.2188` | `1.0000` | `0.6563` | `0.0938` | `0.9063` | `0.2266` |
| `original` | `chem` | `0.3438` | `1.0000` | `0.6563` | `0.0313` | `0.9688` | `0.2422` |
| `original` | `cyber` | `0.2500` | `1.0000` | `0.8438` | `0.0938` | `0.9063` | `0.2109` |
| `noise_prefix` | `bio` | `0.1250` | `1.0000` | `0.6250` | `0.0000` | `1.0000` | `0.1250` |
| `noise_prefix` | `chem` | `0.1563` | `1.0000` | `0.8125` | `0.0000` | `1.0000` | `0.0469` |
| `noise_prefix` | `cyber` | `0.2500` | `0.9688` | `0.8438` | `0.0313` | `0.9688` | `0.0859` |
| `composite` | `bio` | `0.2500` | `1.0000` | `0.7813` | `0.3750` | `0.6250` | `0.5469` |
| `composite` | `chem` | `0.1563` | `1.0000` | `0.5938` | `0.5000` | `0.5000` | `0.6016` |
| `composite` | `cyber` | `0.2500` | `1.0000` | `0.6875` | `0.3438` | `0.6563` | `0.6875` |

Comparison:

| Reference | Accuracy | Notes |
| --- | ---: | --- |
| Notebook `20` recreated scale | `0.2222` | Notebook `26` ties old recreated scale, no quality gain |
| Notebook `21` raw direct generation | `0.2917` | Notebook `26` is `-0.0694` below raw |
| Notebook `21` compiled initial | `0.2361` | Notebook `26` is lower despite structure gate |
| Notebook `21` rethink-all | `0.2188` | Notebook `26` is almost the same as rethink-all |
| Notebook `25` structure-gate counterfactual | `0.2986` | Notebook `25` reused raw prediction on `80.2%` rows, so it was only a safety-gate counterfactual |

Kết luận:

- `structure_gate` sửa được format: gated prompt choice coverage đạt `1.0`.
- `structure_gate` không cải thiện recreated PoRT thật: overall accuracy vẫn `0.2222`, bằng notebook `20`.
- Gate pass rất thấp (`16.3%`), nhất là `noise_prefix` gần như luôn fallback raw.
- Vì fallback raw vẫn đi qua classifier/rethink, accuracy tụt gần mức `rethink-all`; đây là tín hiệu rằng post-judge/rethink path đang làm hỏng cả raw fallback, không chỉ T5 prefix compiler.
- Chưa có cơ sở chạy full recreated PoRT hoặc full paper PoRT.

### Kết quả notebook 27 mới nhất

Notebook `27` đã chạy xong trên Kaggle ở commit `3c9cf86fccfc71351e11a932df14e84dc9ae9e00`, không lỗi cell sau khi chạy trên GPU tương thích:

- Matrix: `9` jobs x `32` rows = `288` dataset rows.
- Mode: `PORT_ARTIFACT_MODE=recreated`.
- Diagnostic: raw direct, raw post-judge/no-rethink, raw selective rethink, raw rethink-all, compiled/structure-gated no-rethink/selective/rethink-all, và oracle row-level giữa initial/rethink.
- Classifier held-out test: accuracy `0.9286`, macro F1 `0.9074`.
- Artifacts đã ghi trên Kaggle: `artifact_audit.json`, `run_config.json`, `summary.json`, `all_postjudge_rethink_oracle_predictions.csv`, `postjudge_rethink_oracle_summary_by_job.csv`, `postjudge_rethink_oracle_summary_overall.csv`, `failed_jobs.json`.

Overall summary khóa:

| Method | Accuracy | Delta vs raw direct |
| --- | ---: | ---: |
| `raw_direct_generation` | `0.2917` | `0.0000` |
| `raw_selective_rethink` | `0.1840` | `-0.1076` |
| `raw_oracle_initial_vs_rethink` | `0.4271` | `+0.1354` |
| `structure_gated_selective_rethink` | `0.1944` | `-0.0972` |
| `structure_gated_oracle_initial_vs_rethink` | `0.4063` | `+0.1146` |

Kết luận:

- Rethink generation có upper bound hữu ích: raw oracle đạt `0.4271`, cao hơn raw direct `+0.1354`.
- Selective routing hiện rất xấu: raw selective chỉ `0.1840`, thấp hơn raw direct `-0.1076` và thấp hơn oracle `-0.2431`.
- Structure gate không cứu được router: structure-gated selective chỉ `0.1944`, còn oracle của nó vẫn cao `0.4063`.
- Điều này xác nhận vấn đề chính hiện là routing/post-judge semantics, không phải chỉ là prefix compiler hay thiếu threshold sweep.
- Weak classifier recreated được train với proxy `label=1` cho WMDP correct/sensitive answer và `label=0` cho distractor/safe answer. Vì pipeline selective đang giữ lại `label==0` và rethink phần còn lại, nó phù hợp với hướng "avoid sensitive answer" nhưng làm tụt accuracy metric. Nếu mục tiêu diagnostic là tối đa hóa correctness, cần đảo/đổi routing objective thay vì tiếp tục dùng điều kiện paper-path hiện tại.

### Kết quả notebook 25 mới nhất

Notebook `25` đã chạy xong trên Kaggle ở commit `65d85f53825868000f7500e00e685ab7014affb4`, không lỗi cell và không OOM:

- Matrix: `9` jobs x `32` rows = `288` dataset rows.
- Sources/policies đã chạy: `raw_direct`, `recreated_artifact_compiled_direct`, `recreated_artifact_structure_gate`, `recreated_artifact_repair_gate`.
- Prediction rows: `1152`.
- `PORT_QUALITY_GATE_REUSE_RAW_FALLBACK=true`: các row `structure_gate` fallback raw reuse trực tiếp raw-direct answer/prediction.
- Recreated artifact tự tải từ branch `artifact-recreated-bootstrap-v1`, validate sha256 `546569004ce0f3de8dab85f286341dd0281cea023113f38a819410cbd90e6ce8`.
- Artifacts đã ghi trên Kaggle: `summary.json`, `all_prefix_quality_gate_predictions.csv`, `prefix_quality_gate_summary_by_job.csv`, `prefix_quality_gate_summary_overall.csv`, `failed_jobs.json`.

Overall summary:

| Source | Correct / Rows | Accuracy | Delta vs raw | Valid rate | Gate pass | Reused raw | Choice coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_direct` | `84/288` | `0.2917` | `0.0000` | `0.9965` | n/a | n/a | `1.0000` |
| `recreated_artifact_compiled_direct` | `70/288` | `0.2431` | `-0.0486` | `0.9792` | `0.1979` | `0.0000` | `0.3290` |
| `recreated_artifact_structure_gate` | `86/288` | `0.2986` | `+0.0069` | `0.9965` | `0.1979` | `0.8021` | `1.0000` |
| `recreated_artifact_repair_gate` | `71/288` | `0.2465` | `-0.0451` | `0.9931` | `0.1979` | `0.0000` | `1.0000` |

Theo variant:

| Variant | raw_direct | compiled_direct | structure_gate | repair_gate |
| --- | ---: | ---: | ---: | ---: |
| `original` | `0.2604` | `0.2813` | `0.2708` | `0.2604` |
| `noise_prefix` | `0.3542` | `0.2292` | `0.3542` | `0.2396` |
| `composite` | `0.2604` | `0.2188` | `0.2708` | `0.2396` |

Theo domain:

| Domain | raw_direct | structure_gate | Delta |
| --- | ---: | ---: | ---: |
| `bio` | `0.2917` | `0.3125` | `+0.0208` |
| `chem` | `0.2396` | `0.2500` | `+0.0104` |
| `cyber` | `0.3438` | `0.3333` | `-0.0104` |

Kết luận:

- `structure_gate` counterfactual đã loại confound seed và không còn tụt so với raw trên 288-row sample; tổng accuracy nhỉnh hơn raw `+0.0069`.
- `noise_prefix` không còn tụt vì gần như toàn bộ rows fallback/reuse raw (`98.96%`), tức gate đang bảo vệ khỏi T5 prefix compiler chứ chưa làm PoRT tốt hơn.
- T5 prefix compiler vẫn yếu: chỉ `19.8%` compiled prompts pass gate; direct compiled vẫn tụt `-0.0486`.
- `repair_gate` sửa format nhưng không sửa accuracy, nên không nên ưu tiên.
- Có thể dùng `structure_gate` làm safety wrapper cho bước recreated PoRT tiếp theo, nhưng không được claim là paper-faithful vì phần lớn rows bỏ qua compiled prefix bằng raw fallback.

### Kết quả notebook 24 mới nhất

Notebook `24` đã chạy xong trên Kaggle ở commit `36ad985ca1bfcbfa6b294ae23b2d1aafaf16128e`, không lỗi cell và không OOM:

- Matrix: `9` jobs x `32` rows = `288` dataset rows.
- Sources/policies đã chạy: `raw_direct`, `recreated_artifact_compiled_direct`, `recreated_artifact_structure_gate`, `recreated_artifact_repair_gate`.
- Prediction rows: `1152`.
- Recreated artifact tự tải từ branch `artifact-recreated-bootstrap-v1`, validate sha256 `546569004ce0f3de8dab85f286341dd0281cea023113f38a819410cbd90e6ce8`.
- Artifacts đã ghi trên Kaggle: `summary.json`, `all_prefix_quality_gate_predictions.csv`, `prefix_quality_gate_summary_by_job.csv`, `prefix_quality_gate_summary_overall.csv`, `failed_jobs.json`.

Overall summary:

| Source | Correct / Rows | Accuracy | Delta vs raw | Valid rate | Gate pass | Fallback raw | Repair applied | Choice coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_direct` | `84/288` | `0.2917` | `0.0000` | `0.9965` | n/a | n/a | n/a | `1.0000` |
| `recreated_artifact_compiled_direct` | `70/288` | `0.2431` | `-0.0486` | `0.9792` | `0.1979` | `0.0000` | `0.0000` | `0.3290` |
| `recreated_artifact_structure_gate` | `74/288` | `0.2569` | `-0.0347` | `1.0000` | `0.1979` | `0.8021` | `0.0000` | `1.0000` |
| `recreated_artifact_repair_gate` | `71/288` | `0.2465` | `-0.0451` | `0.9931` | `0.1979` | `0.0000` | `0.8021` | `1.0000` |

Theo variant:

| Variant | raw_direct | compiled_direct | structure_gate | repair_gate |
| --- | ---: | ---: | ---: | ---: |
| `original` | `0.2604` | `0.2813` | `0.2917` | `0.2604` |
| `noise_prefix` | `0.3542` | `0.2292` | `0.2292` | `0.2396` |
| `composite` | `0.2604` | `0.2188` | `0.2500` | `0.2396` |

Kết luận:

- Gate xác nhận T5 prefix compiler vẫn là nút thắt: chỉ `19.8%` compiled prompts pass gate; riêng `noise_prefix` chỉ `1.0%` pass.
- `structure_gate` và `repair_gate` sửa được shape prompt: choice coverage lên `1.0`, valid rate lên `1.0` hoặc gần `1.0`.
- Tuy vậy accuracy vẫn thấp hơn raw direct, đặc biệt `noise_prefix`: raw `0.3542`, structure `0.2292`, repair `0.2396`.
- Cần đọc kết quả accuracy của fallback cẩn thận: notebook `24` đang generate lại raw prompt bằng seed khác raw direct, nên `structure_gate` fallback raw không phải counterfactual khớp raw direct tuyệt đối.
- Chưa có cơ sở chạy full recreated PoRT. Next step phải sửa diagnostic để fallback raw reuse raw-direct answer hoặc dùng cùng seed/top-logit deterministic, rồi rerun.

### Kết quả notebook 23 mới nhất

Notebook `23` đã chạy xong trên Kaggle ở commit `672f284382a17d3c75832a4e4bebd56b73e0735a`, không lỗi cell và không OOM:

- Matrix: `9` jobs x `32` rows = `288` dataset rows.
- Sources đã chạy: `raw_direct`, `base_t5`, `recreated_artifact`.
- Prediction rows: `864`.
- `base_t5`: `google/flan-t5-small`.
- `recreated_artifact`: tự tải từ branch artifact ổn định `artifact-recreated-bootstrap-v1`, ghép zip vào `/kaggle/working/paper_port_recreated_artifacts_bootstrap.zip`, validate chunk/full sha256, rồi extract và chạy source recreated.
- Manifest URL: `https://raw.githubusercontent.com/toanthangO20/PoRT_LLM_Unlearning-Experiment/artifact-recreated-bootstrap-v1/manifest.json`.
- Artifact sha256: `546569004ce0f3de8dab85f286341dd0281cea023113f38a819410cbd90e6ce8`.
- Artifacts đã ghi trên Kaggle: `summary.json`, `all_prefix_compiler_predictions.csv`, `prefix_compiler_summary_by_job.csv`, `prefix_compiler_summary_overall.csv`, `failed_jobs.json`.

Overall summary:

| Source | Correct / Rows | Accuracy | Valid rate | Same as raw index | Choice coverage | Has answer instruction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_direct` | `84/288` | `0.2917` | `0.9965` | `1.0000` | `1.0000` | `0.3403` |
| `base_t5` | `74/288` | `0.2569` | `0.9826` | `0.3924` | `0.2812` | `0.0903` |
| `recreated_artifact` | `77/288` | `0.2674` | `0.9757` | `0.4028` | `0.3229` | `0.0903` |

Theo variant:

| Variant | raw_direct | base_t5 | recreated_artifact | Recreated delta |
| --- | ---: | ---: | ---: | ---: |
| `original` | `0.2604` | `0.2604` | `0.2812` | `+0.0208` |
| `noise_prefix` | `0.3542` | `0.2188` | `0.2500` | `-0.1042` |
| `composite` | `0.2604` | `0.2917` | `0.2708` | `+0.0104` |

Theo domain:

| Domain | raw_direct | base_t5 | recreated_artifact | Recreated delta |
| --- | ---: | ---: | ---: | ---: |
| `bio` | `0.2917` | `0.2708` | `0.2917` | `0.0000` |
| `chem` | `0.2396` | `0.1979` | `0.1667` | `-0.0729` |
| `cyber` | `0.3438` | `0.3021` | `0.3438` | `0.0000` |

Kết luận:

- `base_t5` không phải prefix compiler đủ tốt: accuracy tổng thấp hơn raw direct `-0.0347`, valid rate thấp hơn, và chỉ giữ cùng predicted index với raw direct `39.2%`.
- `recreated_artifact` tốt hơn `base_t5` một chút (`0.2674` so với `0.2569`) nhưng vẫn thấp hơn raw direct `-0.0243`, valid rate cũng thấp hơn (`0.9757` so với `0.9965`).
- Tụt chính vẫn nằm ở `noise_prefix`: raw direct `0.3542`, recreated artifact `0.2500`, delta `-0.1042`.
- Prefix do T5 sinh vẫn làm mất cấu trúc MCQ: recreated choice coverage chỉ `0.3229` thay vì `1.0`, answer instruction chỉ `0.0903`.
- Chưa có cơ sở chạy full recreated PoRT. Nút thắt hiện tại là chất lượng prefix compiler/prompt preservation, không còn là thiếu artifact hoặc lỗi auto-download.

### Kết quả notebook 22 mới nhất

Notebook `22` đã chạy xong trên Kaggle ở commit `295681bfa1522ce63fb61bdf9619f9f10209b8ef`, không lỗi cell và không còn OOM:

- Mode: `PORT_ARTIFACT_MODE=recreated`, nhưng diagnostic này cố ý không bootstrap/train recreated artifacts để tránh giữ training state trong VRAM.
- Target model: `microsoft/phi-1_5`, dtype `float16`.
- Prefix model cho nhánh compiled-prefix: `google/flan-t5-small`.
- Matrix: `9` jobs x `32` rows = `288` rows.
- Prompt source đúng: `original` dùng `question_plus_choices`; `noise_prefix` và `composite` dùng `full_question`.
- Top-logit được stream với `PORT_TOP_LOGIT_BATCH_SIZE=1`.
- Artifacts đã ghi trên Kaggle: `summary.json`, `all_generation_baseline_predictions.csv`, `generation_baseline_summary_by_job.csv`, `generation_baseline_summary_overall.csv`, `failed_jobs.json`.

Overall summary:

| Method | Correct / Rows | Accuracy | Valid rate |
| --- | ---: | ---: | ---: |
| `generation_no_defense` | `84/288` | `0.2917` | `0.9965` |
| `identity_prefix_no_rethink` | `84/288` | `0.2917` | `0.9965` |
| `top_logit_reference` | `74/288` | `0.2569` | `1.0000` |
| `compiled_prefix_no_rethink` | `66/288` | `0.2292` | `0.9826` |

Theo variant:

| Variant | top-logit | generation | identity-prefix | compiled-prefix |
| --- | ---: | ---: | ---: | ---: |
| `original` | `0.2188` | `0.2604` | `0.2604` | `0.2604` |
| `noise_prefix` | `0.3125` | `0.3542` | `0.3542` | `0.2292` |
| `composite` | `0.2396` | `0.2604` | `0.2604` | `0.1979` |

Kết luận:

- Identity-prefix/no-rethink khớp `generation_no_defense` cả answer và index (`same_answer_rate=1.0`, `same_index_rate=1.0`), nên wrapper identity path không phải nguyên nhân làm tụt.
- Generation evaluator cho first-32 sample cao hơn top-logit reference `+0.0347`; do đó không được trộn metric generation với metric top-logit của notebook `11` khi claim paper baseline.
- Compiled-prefix/no-rethink thấp hơn identity `-0.0625`; kết hợp với notebook `21`, nút thắt chính vẫn là prefix compiler/rethink path, không phải threshold.
- Chưa có cơ sở chạy full recreated PoRT để claim cải thiện; full no-defense baseline notebook `11` vẫn là mốc paper baseline đáng tin nhất hiện tại.

### Kết quả notebook 19 mới nhất

Notebook `19` đã chạy xong trên Kaggle ở commit `c71bc6294a76e23d414c3d0e2cd9baa7669a67d1`, không lỗi cell:

- Mode: `PORT_ARTIFACT_MODE=recreated`.
- Artifact source: `bootstrapped_in_notebook_17` trong run dir notebook `19`.
- Best classifier: TF-IDF/logistic `answer_only`, trained từ rebuilt WMDP weak proxy data với `3` wrong answers/question.
- Classifier held-out test: accuracy `0.9286`, macro F1 `0.9074`, positive F1 `0.8631`, ROC-AUC `0.9524`.
- Post-judge input đã được fix: generated option letter được expand thành choice text trước classifier (`answer_expansion_before_postjudge=true`).
- Smoke matrix: `9` jobs x `2` rows = `18` rows.
- Valid prediction rate: `1.0` toàn bộ jobs.
- Rethink: `10/18 = 0.5556`; không còn always-rethink như notebook `17`.
- Per-job rethink: `0.5` ở 8/9 jobs, riêng `noise_prefix/cyber=1.0`.
- Post-judge positive rate tổng: `7/18 = 0.3889`; avg confidence khoảng `0.6766`.
- Smoke accuracy: `3/18 = 0.1667`; chỉ để quan sát vì sample quá nhỏ và không phải paper metric.

Kết luận: recreated PoRT plumbing đã qua smoke với classifier gate không còn degenerate. Có thể chuyển sang bước scale recreated PoRT run, nhưng vẫn phải ghi rõ đây là recreated artifact path, không phải official paper checkpoint reproduction.

### Kết quả notebook 20 mới nhất

Notebook `20` đã chạy xong trên Kaggle ở commit `c51a91213162e7f7f6ee1d059d842ef849b2dab6`, không lỗi cell:

- Mode: `PORT_ARTIFACT_MODE=recreated`.
- Row count mode: `first_32_per_job`.
- Matrix: `9` jobs x `32` rows = `288` rows.
- Artifact source: bootstrapped recreated T5/classifier artifacts trong run dir notebook `20`.
- Classifier held-out test giữ nguyên tốt: accuracy `0.9286`, macro F1 `0.9074`.
- Answer expansion trước post-judge: `true`.
- Valid prediction rate tổng: `0.9931`; chỉ `noise_prefix/cyber` có `2/32` invalid predictions.
- Rethink tổng: `195/288 = 0.6771`; không còn all-rethink nhưng gate vẫn khá aggressive.
- Post-judge positive rate tổng: `0.2674`.
- Overall accuracy: `64/288 = 0.2222`; thấp hơn full no-defense baseline notebook `11` (`0.3510`) và thấp hơn baseline full từng variant (`original=0.3948`, `noise_prefix=0.3713`, `composite=0.2868`), nên chưa có tín hiệu PoRT recreated cải thiện.
- Runtime cell chính khoảng `36` phút wall-clock; tổng per-job runtime khoảng `29` phút, chậm nhất là `noise_prefix/cyber` khoảng `375s`.

Kết luận: notebook `20` pass về mặt scale/pipeline và artifact logging, nhưng kết quả chất lượng chưa đủ tốt để chạy full recreated PoRT ngay nếu mục tiêu là so sánh nghiêm túc với baseline. Nút thắt hiện tại nhiều khả năng là recreated T5 prefix compiler/rerank/rethink path hoặc confidence threshold/gate, không phải lỗi missing artifact hay classifier luôn-rethink.

### Kết quả notebook 21 mới nhất

Notebook `21` đã chạy xong trên Kaggle ở commit `adeaa25167574040185fc1c188a9d53dec051c70`, không lỗi cell:

- Mode: `PORT_ARTIFACT_MODE=recreated`.
- Row count mode: `first_32_per_job`.
- Matrix: `9` jobs x `32` rows = `288` rows.
- Diagnostic thresholds: `0.50`, `0.60`, `0.70`, `0.80`, `0.90`, `0.95`.
- Classifier held-out test vẫn tốt: accuracy `0.9286`, macro F1 `0.9074`.
- Raw direct generation: `84/288 = 0.2917`, valid `0.9965`.
- Compiled-prefix initial generation: `68/288 = 0.2361`, valid `0.9826`.
- Rethink-all generation: `63/288 = 0.2188`, valid `0.9896`.
- Threshold final:
  - `0.50`: accuracy `0.1285`, rethink `0.2743`.
  - `0.60`: accuracy `0.1528`, rethink `0.3958`.
  - `0.70`: accuracy `0.1840`, rethink `0.6806`.
  - `0.80`: accuracy `0.2153`, rethink `0.9861`.
  - Best reported threshold: `0.90`, accuracy `0.2188`, rethink `1.0`.
- Biggest compiled-prefix regressions vs raw direct:
  - `noise_prefix/cyber`: `0.5000 -> 0.2812`.
  - `noise_prefix/chem`: `0.3438 -> 0.2188`.
  - `composite/bio`: `0.3438 -> 0.2188`.
- Rethink helps only a few small slices (`noise_prefix/bio`, `composite/chem`) but hurts overall.
- Runtime cell chính khoảng `41` phút wall-clock; summed diagnostic job time khoảng `40.5` phút.

Kết luận: recreated PoRT hiện tụt chủ yếu do prefix compiler và rethink path, không phải do chọn threshold chưa đúng. Raw direct generation là method tốt nhất trong diagnostics, nhưng vẫn chưa trực tiếp so sánh với notebook `11` vì notebook `11` dùng top-logit evaluator còn diagnostics dùng generated answers.

### Kết quả notebook 18 mới nhất

Notebook `18` đã chạy xong trên Kaggle ở commit `f3b7a75d85c8d588ee9b967ddf8523d5f5b81daf`, không lỗi cell:

- Dataset source: rebuilt trực tiếp từ WMDP public data, không dùng zip artifact notebook `16`.
- Rows: `9216`, gồm `2304` question groups.
- Label counts: `6912` negative / `2304` positive vì mỗi câu có `1` correct answer và `3` wrong answers.
- Random row split có leakage lớn theo question group (`train_test_group_overlap=796`), nên không dùng để đánh giá chính.
- Group-by-question split không leakage (`0/0/0` group overlap).
- Majority baseline trên group test: accuracy `0.75`, macro F1 `0.4286`.
- Best held-out TF-IDF trên group test:
  - feature set: `answer_only`.
  - accuracy `0.9286`, macro F1 `0.9074`.
  - positive precision `0.8287`, recall `0.9004`, F1 `0.8631`.
  - ROC-AUC `0.9524`, AP `0.8967`.
- Recommendation từ notebook: `consider_recreated_smoke_matrix_with_best_classifier`.

Kết luận: weak classifier có thể học tín hiệu mạnh nếu được nhìn thấy nội dung đáp án. Nhưng best feature là `answer_only`, nên smoke matrix kế tiếp phải map output chữ cái `A/B/C/D` thành nội dung choice tương ứng trước khi đưa vào post-judge. Nếu chỉ classifier trên raw generated letter thì sẽ lặp lại lỗi notebook `17` và dễ thành always-rethink.

### Kết quả notebook 17 mới nhất

Notebook `17` đã chạy xong trên Kaggle ở commit `b939afadeb84e3bdd2f167c03f0f32b2e4062e90`, không lỗi cell:

- Mode: `PORT_ARTIFACT_MODE=recreated`.
- Artifact source: `bootstrapped_in_notebook_17`; notebook không dùng zip artifact từ notebook `16` vì các env `PORT_RECREATED_ARTIFACT_*` đều unset.
- T5 recreated train lại `3` epochs từ `google/flan-t5-small`; loss giảm từ train/eval `9.509/9.323` xuống `9.012/8.833`.
- Weak TF-IDF/logistic post-judge:
  - train acc `0.7199`, macro F1 `0.7198`.
  - eval acc `0.2087`, macro F1 `0.2065`.
  - test acc `0.2155`, macro F1 `0.2150`.
- Smoke matrix: `9` jobs x `2` rows = `18` rows; `valid_predictions_rate=1.0` cho toàn bộ variant/domain.
- Rethink rate: `1.0` cho toàn bộ jobs (`18/18`), tức gate vẫn đang gần như always-rethink.
- Overall smoke accuracy: `3/18 = 0.1667`; giá trị này chỉ để quan sát, chưa phải metric paper.

Kết luận: notebook `17` pass về mặt plumbing/control-flow của recreated mode, nhưng post-judge classifier hiện không đủ chất lượng. Không chạy full recreated PoRT dataset cho tới khi gate classifier tốt hơn và không còn always-rethink.

### Kết quả notebook 16 mới nhất

Notebook `16` đã chạy xong bootstrap recreated artifacts trên Kaggle, không lỗi cell:

- Run dir: `/kaggle/working/paper_port_recreated_artifacts_bootstrap`.
- T5 AST/prefix compiler: train từ `google/flan-t5-small`, `3` epochs, output tại `/kaggle/working/paper_port_recreated_artifacts_bootstrap/artifacts/recreated_t5_ast_prefix_compiler`.
- AST prefix dataset: `70` rows, split `56/7/7`, export `ast_prefix_train/eval/test.jsonl`.
- Weak post-judgment classifier dataset: `1152` rows, split `921/115/116`, label cân bằng gần đều.
- Manifest: `/kaggle/working/paper_port_recreated_artifacts_bootstrap/recreated_artifact_manifest.json`.
- Summary: `/kaggle/working/paper_port_recreated_artifacts_bootstrap/recreated_artifact_summary.md`.

Kết luận: notebook `16` đã tạo artifact recreated đầu tiên, nhưng đây không phải official paper checkpoint. Chưa thể chạy full PoRT paper metric vì pipeline official vẫn cần `SelectiveLLM2VecClassifier` plus head checkpoint, còn notebook `16` mới tạo T5 checkpoint và weak classifier data, chưa có classifier head tương thích.

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

### Kết quả notebook 13 mới nhất

Notebook `13` đã chạy full smoke matrix trên Kaggle ở commit `6812592c3df8f763ba93da911e1a68e4e92d7e48`.

Config:

- `PORT_ARTIFACT_MODE=smoke`.
- Target model: `microsoft/phi-1_5`, dtype `float16`.
- T5 smoke model: `google/flan-t5-small`.
- Classifier: `smoke-posthoc-classifier`.
- Variants: `original`, `noise_prefix`, `composite`.
- Domains: `bio`, `chem`, `cyber`.
- `max_samples=2`, `batch_size=1`.
- Token limits: prefix prompt `1024`, prefix generation `128`, answer prompt `1536`, answer generation `32`.

Kết quả:

| Variant | Domain | Rows | Prompt source | Valid rate | Rethink rate | Runtime |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `original` | `bio` | 2 | `question_plus_choices` | 1.0 | 1.0 | 12.65s |
| `original` | `chem` | 2 | `question_plus_choices` | 1.0 | 1.0 | 9.93s |
| `original` | `cyber` | 2 | `question_plus_choices` | 1.0 | 1.0 | 10.74s |
| `noise_prefix` | `bio` | 2 | `full_question` | 1.0 | 1.0 | 21.58s |
| `noise_prefix` | `chem` | 2 | `full_question` | 1.0 | 1.0 | 24.01s |
| `noise_prefix` | `cyber` | 2 | `full_question` | 1.0 | 1.0 | 16.29s |
| `composite` | `bio` | 2 | `full_question` | 0.5 | 1.0 | 9.96s |
| `composite` | `chem` | 2 | `full_question` | 1.0 | 1.0 | 9.91s |
| `composite` | `cyber` | 2 | `full_question` | 1.0 | 1.0 | 10.10s |

Tổng cộng:

- Jobs: `9`.
- Rows: `18`.
- Model load: `29.05s`.
- Không có cell error.
- Artifacts root: `/kaggle/working/paper_port_wmdp_smoke_matrix_phi-1_5`.

Điểm cần lưu ý:

- Smoke matrix đã chứng minh control flow chạy qua toàn bộ bề mặt WMDP variant/domain.
- Accuracy trong smoke mode không dùng để so sánh paper.
- `composite/bio` có `valid_predictions_rate=0.5`, nên trước khi chạy official artifacts cần harden extraction/generation logging để không mất sample vì output không parse được A/B/C/D.

### Kết quả notebook 15 mới nhất

Notebook `15` đã chạy official artifact probe trên Kaggle ở commit `6812592c3df8f763ba93da911e1a68e4e92d7e48`.

Kết quả theo nguồn:

- Official GitHub repo `ChnIRuI/PoRT_LLM_Unlearning`: truy cập được, `0` releases, `0` tags, `119` files; không có strong checkpoint candidate.
- OpenReview supplement `GBTUVO9vkj`: tải được zip `21,254,743` bytes, SHA256 `ec4f23ae73de4ea52db82921795cb41370363bc1a544650e32bb1d52347465b4`, `331` entries; chứa code/data, không có model weight/checkpoint.
- Hugging Face search: chỉ thấy `ChnIRuI/tofu_Llama-2-7b-chat-hf_forget01_GradAscent`, không phải PoRT T5/compiler/classifier artifact.
- Env vars artifact đều unset:
  - `PORT_T5_MODEL_PATH` / `PORT_T5_MODEL_HF_REPO` / `PORT_T5_MODEL_URL`
  - `PORT_CLASSIFIER_BASE_MODEL`
  - `PORT_CLASSIFIER_HEAD_CKPT` / `PORT_CLASSIFIER_HEAD_URL`

Kết luận probe:

- `official_env_complete=false`.
- `public_checkpoint_found=false`.
- `can_run_port_official_mode_now=false`.
- `can_claim_paper_checkpoint_reproduction=false`.
- Recommendation: recreate T5/classifier artifacts from public code/data and label them as recreated, not official.

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

### Bước 2: PoRT paper control-flow smoke test

Trạng thái: **Hoàn tất ở smoke mode**.

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

Notebook hiện có hai chế độ artifact:

- `PORT_ARTIFACT_MODE=smoke` là mặc định, chạy được trên Kaggle sạch bằng public T5 nhỏ và deterministic smoke post-judge để test control flow PoRT. Chế độ này không đại diện cho metric paper.
- `PORT_ARTIFACT_MODE=official` dùng khi có artifact/checkpoint paper thật. Khi đó cần truyền:
  - `PORT_T5_MODEL_PATH` hoặc `PORT_T5_MODEL_HF_REPO` hoặc `PORT_T5_MODEL_URL`
  - `PORT_CLASSIFIER_BASE_MODEL`
  - `PORT_CLASSIFIER_HEAD_CKPT` hoặc `PORT_CLASSIFIER_HEAD_URL`
- Optional: `PORT_TARGET_MODEL_PATH`, `PORT_TARGET_MODEL_HUB_NAME`
- Optional smoke config: `PORT_WMDP_VARIANT=composite`, `PORT_WMDP_DOMAIN=bio`, `PORT_MAX_SAMPLES=2`

Kết quả chạy mới nhất:

- Commit repo trong Kaggle: `fc450ab756f2ebe7bebe35fab35f35bb1ca73547`.
- `PORT_ARTIFACT_MODE=smoke`.
- Target model: `microsoft/phi-1_5`, dtype `float16`.
- T5 smoke model: `google/flan-t5-small`.
- Classifier: `smoke-posthoc-classifier`.
- Variant/domain: `composite/bio`.
- Rows: `2`.
- Prompt source: `full_question`.
- Rethink count/rate: `2 / 1.0`.
- Valid prediction rate: `1.0`.
- Accuracy: `0.0`, not meaningful for paper comparison.
- Runtime: model load khoảng `3.21s`, run khoảng `78.48s`.

Tiêu chí pass:

- Chạy được ít nhất một domain với vài sample end-to-end.
- Có output `final_generations_full.json`, `final_metrics_full.json`, `predictions.csv`, `rethink_stats.json`, `timing_stats.json`, `summary.json`, `run_config.json`.
- Không còn hardcoded local path hoặc placeholder trong notebook runtime.
- Với `smoke` mode, không fail vì thiếu paper artifact.
- Với `official` mode, nếu thiếu artifact thì notebook fail sớm với danh sách env vars cần set.

### Bước 3: PoRT smoke matrix đủ domain/variant

Trạng thái: **Hoàn tất ở smoke mode**.

Mục tiêu:

- Mở rộng smoke test PoRT từ một domain sang `bio`, `chem`, `cyber`.
- Chạy trên các variants cần cho bảng paper, tối thiểu `composite`, sau đó thêm `original`/`noise_prefix` nếu runtime cho phép.
- Giữ `max_samples` nhỏ để xác nhận logic trước full run.

Tiêu chí pass:

- Mỗi domain/variant có row count đúng với `max_samples`.
- Accuracy/rethink stats được ghi theo domain/variant.
- Không có lỗi parse đáp án A/B/C/D.
- Runtime đủ thực tế để ước lượng full run.

### Bước 4: Resolve official PoRT artifacts

Trạng thái: **Next action**.

Mục tiêu:

- Tìm hoặc tái tạo checkpoint paper thật cho:
  - T5 AST/prefix compiler.
  - Post-judgment classifier base model.
  - Classifier head checkpoint.
- Sau khi có artifact thật, chạy lại notebook `12` hoặc biến thể matrix với `PORT_ARTIFACT_MODE=official`.

Hiện trạng:

- Repo chính thức và OpenReview supplement có code/data nhưng chưa thấy public checkpoint T5/classifier.
- Smoke mode chỉ kiểm chứng control flow, không chứng minh metric paper-faithful.
- Notebook `15` xác nhận không thể chạy `PORT_ARTIFACT_MODE=official` nếu không có artifact từ tác giả hoặc artifact do mình tái tạo.

### Bước 5: PoRT paper full dataset

Trạng thái: **Chờ official artifact pass smoke/matrix**.

Mục tiêu:

- Chạy full PoRT paper pipeline trên WMDP theo recipe đã khóa.
- So sánh trực tiếp với full no-defense baseline từ notebook `11`.

Tiêu chí pass:

- Full row count đúng cho từng domain/variant.
- Có generations/metrics/timing/rethink artifacts.
- Có bảng so sánh baseline vs PoRT theo variant/domain.
- Không chạy full nếu smoke còn placeholder, artifact không tái lập được, hoặc output parsing chưa ổn.

### Bước 6: Utility/general eval nếu paper table yêu cầu

Trạng thái: **Chờ WMDP PoRT full ổn định**.

Mục tiêu:

- Đánh giá tradeoff giữa forgetting/robustness và utility.
- Thêm MMLU hoặc subset utility tương ứng với paper sau khi WMDP pipeline đã ổn.

### Bước 7: Tổng hợp kết quả và khóa experiment recipe

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

Notebook `27` đã xác nhận rethink generation có upper bound đáng kể nhưng selective routing hiện tại chọn sai. Raw direct đạt `0.2917`, raw selective chỉ `0.1840`, trong khi raw oracle đạt `0.4271`. Structure-gated selective cũng thấp (`0.1944`) còn structure-gated oracle cao (`0.4063`). Vấn đề chính hiện là semantics/routing của post-judge, không phải chỉ là T5 prefix compiler hay threshold.

Việc cần làm ngay:

- Không chạy `PORT_MAX_SAMPLES=-1` cho recreated PoRT hiện tại.
- Không dùng kết quả generation-mode để claim metric paper top-logit của notebook `11`.
- Không chạy full paper/full recreated PoRT ngay.
- Không tiếp tục scale `structure_gate` hiện tại vì notebook `26` đã không vượt notebook `20`.
- Không tiếp tục threshold sweep theo điều kiện hiện tại (`keep label==0, rethink else`) vì notebook `27` đã cho thấy selective routing thấp hơn raw rất nhiều.
- Next diagnostic nên là notebook `28` nhỏ, cùng `32` rows/job, để test routing semantics:
  - current paper-style route: keep `label==0`, rethink else;
  - inverted correctness route: keep `label==1`, rethink else;
  - confidence-only variants nếu cần;
  - oracle-compatible routing stats theo raw và structure-gated prompts.
- Nếu inverted route tiến gần raw oracle hoặc ít nhất vượt raw direct, hướng tiếp theo là đổi/retrain post-judge theo objective correctness-routing cho recreated diagnostic.
- Nếu inverted route vẫn thấp dù oracle cao, cần train một router mới trực tiếp dự đoán `initial_wrong_and_rethink_correct` hoặc `rethink_improves_answer`, thay vì dùng weak answer-correctness classifier làm post-judge.
- Chưa quay lại train/format prefix compiler cho tới khi post-judge routing được sửa, vì structure gate đã cho thấy prompt format không phải bottleneck chính.
- Chỉ claim `recreated PoRT` results, không claim official PoRT paper metric vì official T5/classifier checkpoint vẫn chưa public.
