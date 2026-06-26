# LaTeX Beamer slides - LLM Unlearning

Thư mục này chứa project LaTeX Beamer cho bài tập lớn **LLM unlearning cho tri thức nguy hiểm**.

## Cấu trúc

- `main.tex`: file slide chính.
- `beamerthemeHUST.sty`: theme HUST lấy từ template gốc.
- `images/`: ảnh nền và logo của theme HUST.
- `images/eda/`: hình EDA được sinh từ dữ liệu WMDP local.
- `images/papers/`: hình minh họa được crop từ paper gốc dùng trong phần nghiên cứu liên quan.
- `../notebooks/eda/wmdp_eda_for_slides.py`: script tái tạo các hình EDA.

## Build

Trên Overleaf, upload `main.tex`, `beamerthemeHUST.sty` và toàn bộ thư mục `images/`, sau đó chọn compiler `XeLaTeX` để font tiếng Việt hiển thị ổn định nhất.

Nếu máy đã cài TeX Live/MiKTeX:

```powershell
cd E:\PoRT_LLM_Unlearning-Experiment\slide
pdflatex main.tex
pdflatex main.tex
```

Hoặc dùng `latexmk`:

```powershell
cd E:\PoRT_LLM_Unlearning-Experiment\slide
latexmk -pdf main.tex
```

## Cần cập nhật trước khi nộp

- Thay tên thành viên trong `\author{...}`.
- Cập nhật tên giảng viên nếu cần.
- Nếu có kết quả thực nghiệm mới, bổ sung vào phần "Kết quả thực nghiệm và kịch bản đánh giá".
