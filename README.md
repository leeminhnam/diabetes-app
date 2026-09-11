# AI Diabetes Diagnostic System 🩺

Hệ thống hỗ trợ chẩn đoán sớm nguy cơ mắc bệnh tiểu đường bằng Machine Learning. Ứng dụng cung cấp giao diện trực quan cho phép người dùng nhập các chỉ số sức khỏe sinh hóa, từ đó đưa ra dự đoán về nguy cơ mắc bệnh dựa trên 2 mô hình trí tuệ nhân tạo (XGBoost hoặc Deep Learning MLP).

Link Repository: https://github.com/leeminhnam/diabetes-app.git

---

## 🌟 Tính năng nổi bật
* **Chẩn đoán theo thời gian thực:** Phân tích ngay lập tức các chỉ số sức khỏe lâm sàng.
* **Đa Mô hình AI (Multi-Model):** 
  * **Gradient Boosting (XGBoost):** Mô hình dạng cây, cho kết quả có độ chính xác cao và xử lý các đặc trưng phi tuyến tính cực tốt.
  * **Deep Learning (PyTorch MLP):** Mạng nơ-ron sâu đa lớp, mô phỏng cấu trúc nơ-ron thần kinh để tìm ra các mẫu (patterns) phức tạp.
* **Microservice Backend:** Kiến trúc module hóa, mỗi mô hình AI được phân chia thư mục độc lập để dễ dàng bảo trì và scale.
* **Giao diện hiện đại (UI/UX):** Phản hồi thông minh với đồ thị mức độ rủi ro phần trăm (%) và lời khuyên y tế tự động.

---

## ⚙️ Cấu trúc hệ thống
```text
diabetes-app/
├── backend/                  # Python FastAPI Backend
│   ├── app/
│   │   ├── main.py           # Entry point API
│   │   ├── api.py            # API Router
│   │   ├── xgboost/          # XGBoost Microservice
│   │   │   └── service.py    
│   │   └── mlp/              # PyTorch Deep Learning Microservice
│   │       ├── service.py
│   │       ├── model.py      
│   │       └── train_mlp_13.py # Kịch bản huấn luyện mô hình MLP
│   ├── DATA/                 # Thư mục chứa dữ liệu
│   │   └── diabetes.csv      # Dữ liệu gốc
│   └── requirements.txt      # Danh sách thư viện Python
└── frontend/                 # React (Vite) Frontend
    ├── src/
    │   ├── App.jsx           # Giao diện chính
    │   └── App.css           # Styling
    └── package.json          
```

---

## 🚀 Hướng dẫn Cài đặt & Chạy ứng dụng

### 1. Clone Source Code
```bash
git clone https://github.com/leeminhnam/diabetes-app.git
cd diabetes-app
```

### 2. Cài đặt & Chạy Backend (FastAPI)
Yêu cầu: `Python 3.9+`

```bash
# Di chuyển vào thư mục backend
cd backend

# (Tùy chọn) Tạo môi trường ảo
python -m venv venv
# Active môi trường ảo
# Trên Windows: venv\Scripts\activate
# Trên Mac/Linux: source venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt

# Khởi chạy server FastAPI (Server sẽ chạy ở http://127.0.0.1:8000)
uvicorn app.main:app --reload
```

> **Lưu ý:** Nếu bạn cần huấn luyện lại mô hình MLP, hãy chạy lệnh sau:
> `cd backend/app/mlp` sau đó chạy `python train_mlp_13.py`

### 3. Cài đặt & Chạy Frontend (React Vite)
Yêu cầu: `Node.js 18+`

Mở một terminal **mới** (giữ terminal Backend vẫn đang chạy), thực hiện:

```bash
# Di chuyển vào thư mục frontend từ thư mục gốc dự án
cd frontend

# Cài đặt các modules NPM
npm install

# Khởi chạy giao diện React (Client thường sẽ chạy ở http://localhost:5173)
npm run dev
```

---

## 🧠 Các thông số y tế sử dụng trong chẩn đoán
Hệ thống sử dụng các tiêu chí sau (8 đặc trưng) để đánh giá:
1. **Gender:** Giới tính (Nam/Nữ)
2. **Age:** Tuổi
3. **Hypertension:** Tiền sử huyết áp cao (Có/Không)
4. **Heart Disease:** Tiền sử bệnh tim mạch (Có/Không)
5. **Smoking History:** Thói quen hút thuốc lá
6. **BMI:** Chỉ số khối cơ thể (Cân nặng / Chiều cao²)
7. **HbA1c Level:** Chỉ số đường huyết trung bình trong 2-3 tháng
8. **Blood Glucose Level:** Chỉ số đường huyết tức thời

---

## 🛠️ Công nghệ sử dụng
* **Frontend:** ReactJS, Vite, CSS3
* **Backend:** Python, FastAPI, Uvicorn, Pydantic
* **Machine Learning:** Scikit-learn, XGBoost, PyTorch, Pandas, Numpy
