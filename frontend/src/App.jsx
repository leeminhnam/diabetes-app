import React, { useState } from 'react';
import './App.css';

function App() {
  // State lưu trữ 8 trường dữ liệu đầu vào theo đúng tập diabetes.csv
  const [formData, setFormData] = useState({
    gender: 'Female',
    age: 45,
    hypertension: 0,
    heart_disease: 0,
    smoking_history: 'never',
    bmi: 24.5,
    HbA1c_level: 5.7,
    blood_glucose_level: 110,
    model_type: 'xgboost',
  });

  // State phụ trợ giúp tính BMI tự động nếu người dùng không nhớ BMI
  const [height, setHeight] = useState('');
  const [weight, setWeight] = useState('');

  // State quản lý trạng thái API
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Xử lý thay đổi input
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: ['gender', 'smoking_history', 'model_type'].includes(name)
        ? value
        : parseFloat(value) || 0,
    }));
  };

  // Tính BMI tự động từ Chiều cao (cm) & Cân nặng (kg)
  const calculateAutoBMI = () => {
    const h = parseFloat(height);
    const w = parseFloat(weight);
    if (h > 50 && w > 20) {
      const bmiVal = (w / ((h / 100) * (h / 100))).toFixed(1);
      setFormData((prev) => ({ ...prev, bmi: parseFloat(bmiVal) }));
    }
  };

  // Nạp dữ liệu mẫu nhanh để báo cáo/thuyết trình (Demo Preset)
  const loadPreset = (type) => {
    if (type === 'high_risk') {
      setFormData({
        gender: 'Female',
        age: 62,
        hypertension: 1,
        heart_disease: 1,
        smoking_history: 'former',
        bmi: 32.4,
        HbA1c_level: 8.2,
        blood_glucose_level: 210,
      });
    } else {
      setFormData({
        gender: 'Male',
        age: 28,
        hypertension: 0,
        heart_disease: 0,
        smoking_history: 'never',
        bmi: 21.8,
        HbA1c_level: 5.1,
        blood_glucose_level: 95,
      });
    }
    setResult(null);
    setError(null);
  };

  // Gửi request POST đến FastAPI
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Sử dụng đường dẫn tương đối khi build trên Vercel (Production) và localhost khi Dev
      const apiUrl = import.meta.env.PROD ? '/predict' : 'http://127.0.0.1:8000/predict';
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error(`Mã lỗi từ máy chủ: ${response.status}`);
      }

      const data = await response.json();
      if (data.success) {
        setResult(data);
      } else {
        throw new Error(data.error || 'Dự đoán thất bại');
      }
    } catch (err) {
      setError(
        'Không thể kết nối đến Server FastAPI. Nếu đang chạy local, hãy đảm bảo server đang bật ở port 8000!'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-layout">
      {/* Header */}
      <header className="navbar">
        <div className="nav-container">
          <div className="logo-group">
            <span className="logo-icon">🩺</span>
            <div>
              <h2>AI Diabetes Diagnostic</h2>
              <p>Hệ thống hỗ trợ chẩn đoán sớm nguy cơ tiểu đường bằng Machine Learning</p>
            </div>
          </div>
          <div className="preset-buttons">
            <span className="preset-label">Dữ liệu mẫu:</span>
            <button
              type="button"
              className="btn-preset btn-preset-low"
              onClick={() => loadPreset('low_risk')}
            >
              🟢 Mẫu Người Khỏe Mạnh
            </button>
            <button
              type="button"
              className="btn-preset btn-preset-high"
              onClick={() => loadPreset('high_risk')}
            >
              🔴 Mẫu Nguy Cơ Cao
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-grid">
        {/* Form nhập liệu bên trái */}
        <section className="form-card">
          <form onSubmit={handleSubmit}>
            {/* Nhóm 1: Thông tin cá nhân */}
            <div className="section-block">
              <h3 className="section-title">
                <span className="step-num">1</span> Thông tin cơ bản
              </h3>

              <div className="grid-2-col" style={{marginBottom: "15px"}}>
                <div className="input-group">
                  <label>Mô hình AI dự đoán</label>
                  <select name="model_type" value={formData.model_type} onChange={handleChange}>
                    <option value="xgboost">Gradient Boosting (XGBoost)</option>
                    <option value="mlp">Deep Learning (PyTorch MLP)</option>
                  </select>
                </div>
              </div>

              <div className="grid-2-col">
                <div className="input-group">
                  <label>Giới tính</label>
                  <select name="gender" value={formData.gender} onChange={handleChange}>
                    <option value="Female">Nữ (Female)</option>
                    <option value="Male">Nam (Male)</option>
                  </select>
                </div>
                <div className="input-group">
                  <label>Tuổi (năm)</label>
                  <input
                    type="number"
                    name="age"
                    min="1"
                    max="120"
                    value={formData.age}
                    onChange={handleChange}
                    required
                  />
                </div>
              </div>
            </div>

            {/* Nhóm 2: Tiền sử bệnh lý & Thói quen */}
            <div className="section-block">
              <h3 className="section-title">
                <span className="step-num">2</span> Tiền sử bệnh & Lối sống
              </h3>
              <div className="grid-3-col">
                <div className="input-group">
                  <label>Huyết áp cao</label>
                  <select
                    name="hypertension"
                    value={formData.hypertension}
                    onChange={handleChange}
                  >
                    <option value={0}>Không (0)</option>
                    <option value={1}>Có (1)</option>
                  </select>
                </div>
                <div className="input-group">
                  <label>Bệnh tim mạch</label>
                  <select
                    name="heart_disease"
                    value={formData.heart_disease}
                    onChange={handleChange}
                  >
                    <option value={0}>Không (0)</option>
                    <option value={1}>Có (1)</option>
                  </select>
                </div>
                <div className="input-group">
                  <label>Hút thuốc lá</label>
                  <select
                    name="smoking_history"
                    value={formData.smoking_history}
                    onChange={handleChange}
                  >
                    <option value="never">Chưa từng (never)</option>
                    <option value="current">Đang hút (current)</option>
                    <option value="former">Đã cai (former)</option>
                    <option value="not current">Ít khi hút (not current)</option>
                    <option value="No Info">Không rõ (No Info)</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Nhóm 3: Chỉ số xét nghiệm & Thể trạng */}
            <div className="section-block">
              <h3 className="section-title">
                <span className="step-num">3</span> Chỉ số xét nghiệm lâm sàng
              </h3>
              
              {/* Công cụ hỗ trợ tính BMI nhanh */}
              <div className="bmi-helper-box">
                <div className="bmi-helper-inputs">
                  <input
                    type="number"
                    placeholder="Cao (cm)"
                    value={height}
                    onChange={(e) => setHeight(e.target.value)}
                  />
                  <input
                    type="number"
                    placeholder="Nặng (kg)"
                    value={weight}
                    onChange={(e) => setWeight(e.target.value)}
                  />
                  <button type="button" onClick={calculateAutoBMI} className="btn-calc-bmi">
                    ⚡ Tự tính BMI
                  </button>
                </div>
              </div>

              <div className="grid-3-col">
                <div className="input-group">
                  <label>Chỉ số BMI (kg/m²)</label>
                  <input
                    type="number"
                    step="0.1"
                    name="bmi"
                    min="10"
                    max="70"
                    value={formData.bmi}
                    onChange={handleChange}
                    required
                  />
                  <span className="helper-text">Chuẩn: 18.5 - 24.9</span>
                </div>
                <div className="input-group">
                  <label>Chỉ số HbA1c (%)</label>
                  <input
                    type="number"
                    step="0.1"
                    name="HbA1c_level"
                    min="3.0"
                    max="15.0"
                    value={formData.HbA1c_level}
                    onChange={handleChange}
                    required
                  />
                  <span className="helper-text">Bình thường: &lt; 5.7%</span>
                </div>
                <div className="input-group">
                  <label>Đường huyết (mg/dL)</label>
                  <input
                    type="number"
                    name="blood_glucose_level"
                    min="50"
                    max="400"
                    value={formData.blood_glucose_level}
                    onChange={handleChange}
                    required
                  />
                  <span className="helper-text">Lúc đói: 70 - 99 mg/dL</span>
                </div>
              </div>
            </div>

            <button type="submit" className="btn-submit" disabled={loading}>
              {loading ? (
                <span>🔄 Đang phân tích qua mô hình AI...</span>
              ) : (
                <span>🔍 Chạy Chẩn Đoán Ngay</span>
              )}
            </button>
          </form>
        </section>

        {/* Cột hiển thị kết quả bên phải */}
        <section className="result-section">
          {error && (
            <div className="alert-box alert-error">
              <h4>⚠️ Lỗi kết nối</h4>
              <p>{error}</p>
            </div>
          )}

          {!result && !error && !loading && (
            <div className="empty-state">
              <div className="empty-icon">📊</div>
              <h3>Chưa có kết quả phân tích</h3>
              <p>
                Vui lòng điền thông số các chỉ số bên cạnh hoặc chọn <strong>Dữ liệu mẫu</strong> phía trên rồi bấm nút <strong>Chạy Chẩn Đoán</strong>.
              </p>
            </div>
          )}

          {loading && (
            <div className="loading-card">
              <div className="spinner"></div>
              <p>Mô hình Machine Learning đang xử lý vector đặc trưng...</p>
            </div>
          )}

          {result && !loading && (
            <div
              className={`result-card ${
                result.prediction === 1 ? 'card-high-risk' : 'card-low-risk'
              }`}
            >
              <div className="result-header">
                <span className="badge-model">Model: {result.model_used}</span>
                <span className="result-timestamp">Vừa xong</span>
              </div>

              <div className="result-main">
                <div className="result-icon">
                  {result.prediction === 1 ? '🚨' : '🛡️'}
                </div>
                <div>
                  <h2 className="result-status">{result.diagnosis}</h2>
                  <p className="result-desc">
                    {result.prediction === 1
                      ? 'Các chỉ số sinh hóa và tiền sử cho thấy tỷ lệ nguy cơ tiểu đường ở mức đáng báo động.'
                      : 'Các chỉ số sức khỏe của bạn hiện tại nằm trong ngưỡng an toàn ổn định.'}
                  </p>
                </div>
              </div>

              {/* Thanh đo % nguy cơ */}
              <div className="meter-container">
                <div className="meter-labels">
                  <span>Mức độ nguy cơ:</span>
                  <strong className="risk-percent">{result.risk_percentage}%</strong>
                </div>
                <div className="meter-bar-bg">
                  <div
                    className="meter-bar-fill"
                    style={{
                      width: `${result.risk_percentage}%`,
                      backgroundColor:
                        result.prediction === 1 ? '#ef4444' : '#10b981',
                    }}
                  ></div>
                </div>
              </div>

              {/* Lời khuyên y tế */}
              <div className="advice-box">
                <h4>💡 Khuyến nghị y tế:</h4>
                <ul>
                  {result.prediction === 1 ? (
                    <>
                      <li>Nên đến cơ sở y tế gần nhất để làm xét nghiệm dung nạp Glucose và đo lại HbA1c tĩnh mạch.</li>
                      <li>Hạn chế tối đa tinh bột hấp thu nhanh, đường tinh luyện và đồ ngọt đóng chai.</li>
                      <li>Duy trì ít nhất 30 phút đi bộ hoặc vận động nhẹ mỗi ngày.</li>
                    </>
                  ) : (
                    <>
                      <li>Tiếp tục duy trì chế độ dinh dưỡng lành mạnh và cân đối.</li>
                      <li>Kiểm tra sức khỏe định kỳ mỗi 6 - 12 tháng một lần.</li>
                      <li>Duy trì cân nặng và chỉ số BMI trong khoảng 18.5 - 23 kg/m².</li>
                    </>
                  )}
                </ul>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;