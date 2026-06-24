# Báo cáo tổng quan: LLM unlearning cho tri thức nguy hiểm

Ngày tạo: 2026-06-25

## Phạm vi báo cáo

Báo cáo này tạm thời tách khỏi các thử nghiệm cụ thể trong repository hiện tại và tập trung vào bức tranh tổng quát của bài toán **LLM unlearning cho tri thức nguy hiểm**. Trọng tâm là bối cảnh đánh giá bằng bộ dữ liệu **WMDP - Weapons of Mass Destruction Proxy**, một benchmark được dùng rộng rãi để đo lường và giảm năng lực trả lời của mô hình về các miền tri thức có khả năng bị lạm dụng như an toàn sinh học, an ninh mạng và an toàn hóa học.

Vì chủ đề có liên quan đến tri thức nguy hiểm, báo cáo không trích nguyên văn các câu hỏi nhạy cảm trong WMDP. Ví dụ record ở phần dữ liệu là ví dụ minh họa tổng hợp, giữ đúng cấu trúc dữ liệu nhưng không chứa nội dung vận hành nguy hiểm.

## 1. Đặt vấn đề

Các mô hình ngôn ngữ lớn được huấn luyện trên kho dữ liệu rất rộng, nhờ đó có khả năng trả lời câu hỏi chuyên sâu trong nhiều lĩnh vực. Lợi ích này đi kèm rủi ro: cùng một năng lực suy luận và truy hồi tri thức có thể hỗ trợ người dùng trong các miền nhạy cảm như sinh học, hóa học, an ninh mạng hoặc các kỹ thuật có khả năng bị lạm dụng.

Các cơ chế an toàn thông thường, như instruction tuning, RLHF, bộ lọc đầu vào/đầu ra hoặc refusal policy, chủ yếu kiểm soát **hành vi sinh câu trả lời**. Tuy nhiên, chúng không nhất thiết loại bỏ tri thức khỏi mô hình. Khi gặp prompt được diễn đạt khác, jailbreak, fine-tuning lại, hoặc truy vấn gián tiếp, mô hình có thể bộc lộ lại tri thức mà hệ thống mong muốn hạn chế. Do đó, bài toán LLM unlearning đặt câu hỏi khó hơn: làm thế nào để làm giảm hoặc loại bỏ ảnh hưởng của một tập tri thức không mong muốn trong mô hình, trong khi vẫn giữ được năng lực hữu ích ở các miền còn lại.

Trong bối cảnh **tri thức nguy hiểm**, mục tiêu không chỉ là bảo vệ quyền riêng tư hoặc tuân thủ yêu cầu xóa dữ liệu, mà còn là giảm khả năng mô hình hỗ trợ các hành vi gây hại. Đây là một bài toán vừa kỹ thuật vừa đánh giá an toàn: nếu giảm quá ít, mô hình vẫn rủi ro; nếu giảm quá mạnh, mô hình mất năng lực khoa học, y sinh, bảo mật phòng thủ hoặc tri thức phổ thông hợp pháp.

## 2. Định nghĩa bài toán

### 2.1 Mô hình hóa tổng quát

Giả sử có một mô hình ngôn ngữ lớn ban đầu:

```text
M_theta
```

Ta có hai nhóm dữ liệu hoặc miền tri thức:

```text
D_f: forget set, gồm tri thức/nhiệm vụ cần làm quên hoặc giảm ảnh hưởng
D_r: retain set, gồm tri thức/nhiệm vụ cần giữ lại
```

Mục tiêu là tạo ra một mô hình hoặc một hệ thống sau unlearning:

```text
M'
```

sao cho:

```text
Performance(M', D_f) giảm theo tiêu chí an toàn
Performance(M', D_r) được giữ càng gần M_theta càng tốt
M' bền vững trước paraphrase, jailbreak, prompt attack, relearning/fine-tuning attack
```

Trong bài toán WMDP, `D_f` thường tương ứng với các miền tri thức nguy hiểm như biosecurity, cybersecurity, chemical security. `D_r` thường là các benchmark năng lực chung hoặc lân cận như MMLU, Wikitext, SciQ hoặc các tập retain chuyên biệt.

### 2.2 Đầu vào

Đầu vào của bài toán có thể gồm:

- **Mô hình gốc**: trọng số của LLM hoặc quyền truy cập inference API.
- **Forget specification**: tập câu hỏi, tài liệu, mẫu prompt hoặc mô tả miền tri thức cần làm quên.
- **Retain specification**: tập dữ liệu hoặc benchmark dùng để đo năng lực cần giữ.
- **Ràng buộc triển khai**: có/không có quyền sửa trọng số; chi phí tính toán; yêu cầu chạy inference-time hay training-time; mức truy cập white-box/black-box.

Với WMDP, một mẫu đánh giá thường là câu hỏi trắc nghiệm:

```text
x = (question, choices)
y = answer index
```

trong đó mô hình được yêu cầu chọn đáp án đúng. Accuracy trên WMDP càng thấp sau unlearning thường được xem là càng quên tốt, nhưng phải được diễn giải cùng với accuracy trên retain set.

### 2.3 Đầu ra

Đầu ra của một phương pháp unlearning có thể là:

- **Mô hình đã cập nhật trọng số**: ví dụ unlearning bằng gradient, representation steering, weight attribution.
- **Adapter hoặc delta weights**: thay đổi cục bộ/nhẹ trên một phần tham số.
- **Cơ chế inference-time**: bộ phân loại prompt, soft prompt, embedding corruption hoặc router.
- **Báo cáo đánh giá**: gồm forget efficacy, retain utility, robustness, chi phí và phân tích lỗi.

Trong bối cảnh tri thức nguy hiểm, đầu ra không nên chỉ được đánh giá bằng một con số accuracy. Một hệ thống có vẻ "quên" trên benchmark tĩnh nhưng vẫn dễ bị khôi phục bằng prompt khác hoặc relearning attack thì chưa đạt yêu cầu an toàn thực tế.

## 3. Những thách thức chính

### 3.1 Ranh giới giữa tri thức nguy hiểm và tri thức hợp pháp không rõ ràng

Cùng một mảng kiến thức có thể hữu ích cho nghiên cứu, phòng thủ, y tế, giáo dục hoặc kiểm thử an toàn, nhưng cũng có thể bị dùng sai mục đích. Ví dụ, kiến thức sinh học hoặc an ninh mạng không thể bị xóa toàn bộ, vì điều đó làm suy giảm năng lực hợp pháp của mô hình.

### 3.2 Tri thức trong LLM có tính phân tán và đan xen

LLM không lưu tri thức như một bảng tra cứu tách biệt. Một tri thức nguy hiểm có thể được mã hóa qua nhiều lớp, neuron, attention head và biểu diễn trung gian. Do đó, xóa một miền tri thức thường gây tác động phụ lên các miền gần kề.

### 3.3 Không có "mô hình lý tưởng đã quên" để so sánh

Trong lý thuyết machine unlearning, mô hình lý tưởng là mô hình được retrain từ đầu mà không có dữ liệu cần quên. Với LLM, retrain từ đầu gần như không khả thi vì chi phí rất lớn và thường không có toàn bộ dữ liệu gốc. Vì vậy, hầu hết phương pháp chỉ là approximate unlearning.

### 3.4 Đánh giá bằng accuracy có thể gây hiểu nhầm

Với WMDP, accuracy thấp trên câu hỏi trắc nghiệm cho thấy mô hình trả lời kém hơn trên benchmark. Tuy nhiên, điều này chưa đủ để chứng minh tri thức đã bị xóa. Mô hình có thể vẫn chứa tri thức nhưng bị làm nhiễu ở định dạng câu hỏi cụ thể, bị lệch chọn đáp án, hoặc bị ép trả lời sai mà vẫn có thể sinh thông tin khi đổi cách hỏi.

### 3.5 Robustness là yêu cầu trung tâm

Một phương pháp unlearning cần chịu được:

- Paraphrase và thay đổi ngữ cảnh truy vấn.
- Jailbreak hoặc prompt injection.
- Relearning attack, tức fine-tune lại bằng rất ít mẫu quên.
- Truy vấn gián tiếp hoặc multi-hop.
- Khác biệt giữa benchmark tĩnh và hành vi đối thoại thực tế.

### 3.6 Over-forgetting và utility loss

Nếu phương pháp làm giảm quá mạnh biểu diễn của một miền, mô hình có thể mất cả tri thức hợp pháp lân cận. Ví dụ, unlearning WMDP-Bio không nên làm mô hình mất năng lực sinh học phổ thông hoặc y sinh an toàn.

## 4. Bộ dữ liệu WMDP

### 4.1 Vai trò của WMDP

WMDP được giới thiệu trong công trình **The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning** tại ICML 2024. Theo bài báo, WMDP gồm **3,668 câu hỏi trắc nghiệm** dùng làm proxy để đo tri thức nguy hiểm trong ba miền: biosecurity, cybersecurity và chemical security. WMDP có hai vai trò:

- Benchmark đánh giá mức độ mô hình nắm tri thức nguy hiểm.
- Benchmark cho các phương pháp unlearning nhằm giảm năng lực này.

Nguồn chính:

- Paper ICML 2024: <https://proceedings.mlr.press/v235/li24bc.html>
- Dataset card Hugging Face: <https://huggingface.co/datasets/cais/wmdp>
- Repository WMDP: <https://github.com/centerforaisafety/wmdp>

### 4.2 Các subset

WMDP thường được chia theo ba miền:

| Subset | Miền đánh giá | Vai trò |
|---|---|---|
| `wmdp-bio` | Biosecurity | Đánh giá tri thức sinh học có khả năng bị lạm dụng |
| `wmdp-cyber` | Cybersecurity | Đánh giá tri thức an ninh mạng nhạy cảm |
| `wmdp-chem` | Chemical security | Đánh giá tri thức hóa học nhạy cảm |

Lưu ý: số lượng mẫu theo từng subset có thể khác nhau giữa các phiên bản/mirror do WMDP từng được cập nhật để sửa định dạng, Unicode và loại bỏ một số câu hỏi. Khi viết báo cáo thực nghiệm, cần ghi rõ commit/version dataset được dùng.

### 4.3 Schema dữ liệu

Schema lõi của WMDP dạng Hugging Face dataset/parquet thường gồm:

```json
{
  "question": "string",
  "choices": ["string", "string", "string", "string"],
  "answer": "int64"
}
```

Ý nghĩa các trường:

| Trường | Kiểu dữ liệu | Mô tả |
|---|---|---|
| `question` | `string` | Câu hỏi trắc nghiệm |
| `choices` | `list[string]` | Danh sách bốn lựa chọn trả lời |
| `answer` | `int64` | Chỉ số đáp án đúng, thường là 0-3 tương ứng A-D |

Trong một số pipeline hoặc biến thể dữ liệu, có thể xuất hiện thêm các trường đã xử lý sẵn như:

| Trường phụ | Mô tả |
|---|---|
| `full_question` | Prompt/câu hỏi đã được format đầy đủ với lựa chọn A-D |
| `noisy_prefix` | Prefix nhiễu dùng cho biến thể đánh giá hoặc pipeline đặc biệt |
| `harmful_position` | Vị trí phần nội dung cần kiểm soát trong biến thể composite |

Các trường phụ này không phải schema lõi của WMDP gốc, mà thường là artifact của pipeline hoặc biến thể thực nghiệm.

### 4.4 Định dạng dữ liệu

WMDP được phân phối phổ biến qua Hugging Face Dataset, có thể ở dạng parquet hoặc Arrow sau khi tải về. Khi đưa vào pipeline đánh giá, record thường được format thành prompt trắc nghiệm:

```text
Question: ...
A. ...
B. ...
C. ...
D. ...
Answer:
```

Mô hình có thể được chấm bằng:

- Logit trên các token A/B/C/D.
- Sinh câu trả lời rồi parse đáp án.
- So sánh lựa chọn cuối cùng với `answer`.

### 4.5 Ví dụ một record minh họa an toàn

Ví dụ dưới đây là record tổng hợp, không phải câu hỏi thật trong WMDP:

```json
{
  "question": "Trong một quy trình đánh giá an toàn AI, bước nào phù hợp nhất để giảm nguy cơ mô hình cung cấp tri thức nguy hiểm?",
  "choices": [
    "Đánh giá truy vấn theo miền rủi ro trước khi sinh câu trả lời",
    "Tăng số token sinh ra cho mọi truy vấn",
    "Bỏ qua kiểm thử sau huấn luyện",
    "Công bố mọi đầu ra không qua kiểm duyệt"
  ],
  "answer": 0
}
```

Record này giữ đúng cấu trúc `question`, `choices`, `answer` nhưng không chứa nội dung nguy hiểm vận hành.

## 5. Các hướng phương pháp chính

### 5.1 Weight-level unlearning

Nhóm này cập nhật trực tiếp trọng số mô hình. Ví dụ:

- Gradient ascent trên forget set.
- Gradient difference giữa forget loss và retain loss.
- Preference optimization hoặc negative preference optimization.
- Representation misdirection, như RMU.
- Weight attribution hoặc sửa một phần tham số.

Ưu điểm là có khả năng làm thay đổi nội tại mô hình. Hạn chế là cần truy cập trọng số, tốn tài nguyên và dễ gây utility loss.

### 5.2 Representation-level unlearning

Nhóm này can thiệp vào biểu diễn ẩn, thường ở một số layer nhất định. RMU là đại diện quan trọng: thay vì chỉ tối ưu loss đầu ra, phương pháp hướng biểu diễn của dữ liệu cần quên sang một vector mục tiêu không mang thông tin hữu ích, đồng thời giữ biểu diễn của dữ liệu retain.

Ưu điểm là đánh trực tiếp vào biểu diễn trung gian. Hạn chế là nhạy với layer, hệ số steering, retain set và mức đan xen giữa forget/retain.

### 5.3 Inference-time unlearning

Nhóm này không nhất thiết sửa trọng số mô hình, mà áp dụng cơ chế khi inference:

- Prompt classifier để nhận diện truy vấn thuộc miền cần quên.
- Soft prompt hoặc prefix học được.
- Corruption trong embedding space.
- Router chọn nhánh trả lời an toàn.

Ưu điểm là nhẹ, áp dụng được cả với API hoặc mô hình đóng. Hạn chế là giống guardrail hơn là xóa tri thức khỏi mô hình; độ an toàn phụ thuộc vào classifier/router và khả năng chống né tránh.

### 5.4 Robust unlearning

Nhóm này tập trung vào việc làm cho unlearning bền vững hơn trước jailbreak, adversarial suffix, paraphrase hoặc relearning attack. Đây là hướng rất quan trọng vì nhiều phương pháp có thể đạt điểm tốt trên benchmark tĩnh nhưng thất bại khi bị tấn công có chủ đích.

## 6. Nghiên cứu liên quan có thực nghiệm trên WMDP

Bảng dưới đây chỉ chọn các công trình đã công bố tại hội nghị uy tín và có thực nghiệm trên WMDP hoặc subset của WMDP. Đây không phải danh sách đầy đủ, nhưng là nhóm công trình tiêu biểu để xây dựng phần related work.

| Công trình | Hội nghị | Cách dùng WMDP | Ý tưởng chính | Hạn chế/khoảng trống |
|---|---:|---|---|---|
| **The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning** | ICML 2024 | Giới thiệu WMDP và dùng WMDP để đánh giá RMU | Xây dựng benchmark hazardous knowledge và đề xuất RMU/representation misdirection | Accuracy thấp trên WMDP chưa chứng minh tri thức bị xóa hoàn toàn; robustness và khả năng khôi phục tri thức vẫn là vấn đề |
| **Large Language Model Unlearning via Embedding-Corrupted Prompts** | NeurIPS 2024 | Đánh giá trên WMDP/MMLU trong bối cảnh unlearning | ECO Prompts dùng prompt classifier và corruption trong embedding space ở inference time | Phụ thuộc vào classifier; thiên về kiểm soát truy cập/hành vi hơn là xóa trọng số |
| **WAGLE: Strategic Weight Attribution for Effective and Modular Unlearning in Large Language Models** | NeurIPS 2024 | Dùng WMDP như benchmark malicious-use prevention | Xác định trọng số có ảnh hưởng đến forget/retain để hướng dẫn unlearning mô-đun hơn | Weight attribution khó trong mô hình lớn; cần truy cập trọng số và có thể nhạy với mô hình/dữ liệu |
| **On Effects of Steering Latent Representation for Large Language Model Unlearning** | AAAI 2025 | Thực nghiệm trên WMDP-Biology, WMDP-Cyber và MMLU | Phân tích cơ chế RMU và đề xuất Adaptive RMU | Nhạy với layer/hyperparameter; chủ yếu ở mô hình mở cỡ 7B; retain set gần miền forget có thể làm unlearning khó hơn |
| **Soft Prompting for Unlearning in Large Language Models** | NAACL 2025 | Có thí nghiệm MCQA trên WMDP+SciQ | Học soft prompt để tạo trạng thái unlearning mà không cập nhật trọng số gốc | Không xóa tri thức khỏi base model; phụ thuộc phân phối prompt và cách gắn soft prompt |
| **SEUF: Is Unlearning One Expert Enough for Mixture-of-Experts LLMs?** | ACL 2025 | Dùng WMDP và RWKU để đánh giá unlearning trên MoE LLMs | Nghiên cứu unlearning trong mô hình Mixture-of-Experts và chọn expert/tham số cần can thiệp | Kết luận phụ thuộc kiến trúc MoE; chưa giải quyết toàn bộ bài toán robust unlearning cho mọi kiểu mô hình |
| **Towards LLM Unlearning Resilient to Relearning Attacks** | ICML 2025 | Thực nghiệm trên WMDP và MUSE | Liên hệ robust unlearning với sharpness-aware minimization; chống relearning attack | Tăng chi phí tối ưu; robustness chỉ được kiểm chứng trong một số dạng tấn công và mô hình |
| **Exploring Criteria of Loss Reweighting to Enhance LLM Unlearning** | ICML 2025 | Đánh giá tiêu chí loss reweighting trên WMDP cùng các benchmark unlearning khác | Đề xuất cách reweight loss theo saturation/importance để cải thiện trade-off forget-retain | Vẫn phụ thuộc thiết kế loss, retain data và lựa chọn tiêu chí; không giải quyết triệt để vấn đề định nghĩa "đã quên" |

Nguồn tham khảo chính cho các công trình trong bảng:

- WMDP/RMU, ICML 2024: <https://proceedings.mlr.press/v235/li24bc.html>
- ECO Prompts, NeurIPS 2024: <https://proceedings.neurips.cc/paper_files/paper/2024/hash/d6359156e0e30b1caa116a4306b12688-Abstract-Conference.html>
- WAGLE, NeurIPS 2024: <https://proceedings.neurips.cc/paper_files/paper/2024/hash/649ad92e7067b3553a0f15acac68806d-Abstract-Conference.html>
- Adaptive RMU, AAAI 2025: <https://ojs.aaai.org/index.php/AAAI/article/view/34544>
- SPUL, NAACL 2025: <https://aclanthology.org/2025.naacl-long.204/>
- SEUF, ACL 2025: <https://aclanthology.org/2025.acl-long.424/>
- Robust relearning attacks, ICML 2025: <https://proceedings.mlr.press/v267/fan25e.html>
- Loss reweighting/SatImp, ICML 2025: <https://openreview.net/forum?id=mGOugCZlAq>

## 7. Hạn chế chung của các nghiên cứu hiện có

### 7.1 Chưa phân biệt rõ "không trả lời" và "đã quên"

Nhiều phương pháp làm giảm accuracy hoặc khiến mô hình từ chối trả lời, nhưng điều đó chưa chứng minh tri thức đã bị xóa khỏi tham số. Đặc biệt, các phương pháp inference-time có thể che hành vi ở lớp ngoài nhưng tri thức vẫn còn trong base model.

### 7.2 Benchmark tĩnh chưa đủ cho an toàn thực tế

WMDP là benchmark quan trọng vì công khai và có độ chuyên môn cao, nhưng vẫn là tập câu hỏi trắc nghiệm hữu hạn. Một phương pháp có thể overfit vào định dạng WMDP, trong khi prompt thực tế có thể dài hơn, gián tiếp hơn hoặc chứa nhiều bước suy luận.

### 7.3 Robustness chưa được chuẩn hóa

Các công trình bắt đầu đánh giá jailbreak, adversarial suffix, relearning attack hoặc paraphrase, nhưng chưa có giao thức thống nhất. Điều này khiến kết quả giữa các paper khó so sánh trực tiếp.

### 7.4 Retain set quyết định kết luận nhưng thường bị xem nhẹ

Nếu retain set quá xa forget domain, mô hình có thể giữ utility chung nhưng vẫn mất tri thức hợp pháp lân cận. Nếu retain set quá gần forget domain, unlearning lại trở nên khó do tri thức đan xen. Việc chọn retain set là một phần cốt lõi của bài toán, không chỉ là chi tiết thực nghiệm.

### 7.5 Thiếu thước đo cơ chế nội tại

Accuracy, refusal rate hoặc logit score chỉ phản ánh hành vi đầu ra. Cần thêm phân tích biểu diễn, probing, causal tracing, khả năng relearning và kiểm thử truy hồi gián tiếp để đánh giá liệu tri thức đã bị loại bỏ hay chỉ bị che.

## 8. Khoảng trống nghiên cứu

Từ các công trình trên, có thể rút ra các khoảng trống chính:

1. **Định nghĩa operational về "unlearned hazardous knowledge"**: cần phân biệt giảm năng lực trả lời benchmark, giảm xác suất sinh nội dung nguy hiểm và xóa ảnh hưởng tri thức khỏi mô hình.

2. **Đánh giá robustness toàn diện**: cần benchmark kết hợp WMDP với paraphrase, jailbreak, multi-turn, retrieval-augmented prompting và relearning attack.

3. **Giữ tri thức hợp pháp lân cận**: unlearning không nên làm mất năng lực sinh học/phòng thủ mạng/hóa học an toàn. Cần đánh giá retain set cùng miền nhưng không nguy hiểm.

4. **Giảm phụ thuộc vào classifier hoặc router thủ công**: các cơ chế inference-time hiệu quả nhưng dễ tạo điểm yếu nếu bộ nhận diện prompt sai hoặc bị né tránh.

5. **Phương pháp có khả năng mở rộng cho mô hình đóng**: nhiều kỹ thuật mạnh cần truy cập trọng số, trong khi thực tế nhiều LLM quan trọng chỉ truy cập qua API.

6. **Thước đo vượt ngoài accuracy**: cần kết hợp accuracy, calibration, refusal quality, semantic leakage, activation analysis, relearning cost và adversarial robustness.

7. **An toàn dữ liệu và công bố benchmark**: benchmark phải đủ đại diện để đánh giá rủi ro, nhưng không được vô tình phát tán hướng dẫn nguy hiểm. Đây là căng thẳng cố hữu của WMDP và các benchmark tương tự.

## 9. Kết luận ngắn

LLM unlearning cho tri thức nguy hiểm là bài toán giảm năng lực mô hình trong các miền có thể bị lạm dụng mà vẫn giữ lại tri thức hợp pháp và năng lực chung. WMDP hiện là benchmark trung tâm vì cung cấp bộ câu hỏi công khai, có cấu trúc rõ ràng và gắn trực tiếp với mục tiêu malicious-use reduction.

Tuy nhiên, kết quả tốt trên WMDP không tự động đồng nghĩa với việc mô hình đã thật sự quên. Các hướng nghiên cứu gần đây đang dịch chuyển từ tối ưu accuracy sang đánh giá cơ chế, giữ utility lân cận và chống khôi phục tri thức qua jailbreak hoặc relearning. Khoảng trống nghiên cứu quan trọng nhất nằm ở việc xây dựng phương pháp unlearning vừa **hiệu quả**, **bền vững**, **ít tác dụng phụ**, vừa có **bằng chứng đánh giá thuyết phục** rằng tri thức nguy hiểm không chỉ bị che ở đầu ra mà thực sự khó truy hồi trong nhiều điều kiện sử dụng.

