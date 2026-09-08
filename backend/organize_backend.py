import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(BASE_DIR, 'app')
XGB_DIR = os.path.join(APP_DIR, 'xgboost')
MLP_DIR = os.path.join(APP_DIR, 'mlp')
ML_MODELS_DIR = os.path.join(BASE_DIR, 'ml_models')

# Đảm bảo các thư mục đích tồn tại
os.makedirs(XGB_DIR, exist_ok=True)
os.makedirs(MLP_DIR, exist_ok=True)

# 1. Các file dùng chung: copy vào cả 2 thư mục (Microservice pattern)
shared_files = ['scaler_diabetes.pkl', 'model_columns.json']
for f in shared_files:
    src_path = os.path.join(ML_MODELS_DIR, f)
    if not os.path.exists(src_path):
        src_path = os.path.join(BASE_DIR, f) # Fallback
    if os.path.exists(src_path):
        shutil.copy2(src_path, os.path.join(XGB_DIR, f))
        shutil.copy2(src_path, os.path.join(MLP_DIR, f))
        print(f"✅ Đã copy {f} vào xgboost/ và mlp/")

# 2. File của XGBoost
xgb_files = {
    'best_diabetes_model.pkl': os.path.join(ML_MODELS_DIR, 'best_diabetes_model.pkl'),
    'diabetes_prediction.ipynb': os.path.join(BASE_DIR, 'diabetes_prediction.ipynb')
}
for name, src_path in xgb_files.items():
    if not os.path.exists(src_path) and os.path.exists(os.path.join(BASE_DIR, name)):
        src_path = os.path.join(BASE_DIR, name)
    if os.path.exists(src_path):
        shutil.copy2(src_path, os.path.join(XGB_DIR, name))
        print(f"✅ Đã copy {name} vào xgboost/")

# 3. File của MLP
mlp_files = {
    'mlp_pytorch_experiments.ipynb': os.path.join(BASE_DIR, 'mlp_pytorch_experiments.ipynb'),
}
for name, src_path in mlp_files.items():
    if os.path.exists(src_path):
        shutil.copy2(src_path, os.path.join(MLP_DIR, name))
        print(f"✅ Đã copy {name} vào mlp/")

# 4. Xóa các file thừa bên ngoài
files_to_remove = [
    'best_diabetes_model.pkl',
    'scaler_diabetes.pkl',
    'model_columns.json',
    'diabetes_model_columns.json',
    'diabetes_prediction.ipynb',
    'mlp_pytorch_experiments.ipynb',
    'train_mlp_13.py',
    'app/ml_service.py',
    'app/mlp_model.py'
]

print("\n--- DỌN DẸP FILE THỪA ---")
for f in files_to_remove:
    path = os.path.join(BASE_DIR, f)
    if os.path.exists(path):
        try:
            os.remove(path)
            print(f"🗑️ Đã xóa: {f}")
        except Exception as e:
            print(f"❌ Không thể xóa {f}: {e}")

# Xóa folder ml_models cũ
if os.path.exists(ML_MODELS_DIR):
    try:
        shutil.rmtree(ML_MODELS_DIR)
        print(f"🗑️ Đã xóa thư mục: ml_models/")
    except Exception as e:
        print(f"❌ Không thể xóa thư mục ml_models: {e}")

print("\n🎉 Hoàn thành tổ chức lại thư mục backend!")
print("Cấu trúc mới:")
print("backend/app/")
print("  ├── xgboost/")
print("  │   ├── best_diabetes_model.pkl")
print("  │   ├── scaler_diabetes.pkl")
print("  │   ├── model_columns.json")
print("  │   └── ...")
print("  └── mlp/")
print("      ├── scaler_diabetes.pkl")
print("      ├── model_columns.json")
print("      ├── train_mlp_13.py")
print("      └── ...")
