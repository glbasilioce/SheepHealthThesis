# Sheep Health Monitoring System

AI-based disease detection system for sheep with integrated health monitoring.

## 🎯 Model Performance

- **Overall Accuracy:** 97.96%
- **Test Results:** 144/147 correct predictions
- **Architecture:** MobileNetV2 (Transfer Learning)
- **Training Method:** Stratified split validation

### Per-Class Accuracy
- **Healthy:** 100% (43/43) 🏆
- **Orf:** 100% (34/34) 🏆
- **Sheep Scab:** 100% (39/39) 🏆
- **Flystrike:** 90.32% (28/31)

## 📊 Classes
1. **Flystrike** (Myiasis) - Fly larvae infestation
2. **Healthy** - No disease detected
3. **Orf** (Contagious Ecthyma) - Viral infection
4. **Sheep Scab** (Psoroptic Mange) - Mite infestation

## 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/glbasilioce/SheepHealthThesis.git
cd SheepHealthThesis

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
python run.py
```

Open browser: `http://localhost:5000`

## 📁 Key Files

- `models/sheep_disease_4class_stratified.keras` - Trained AI model (97.96% accuracy)
- `thesis_metrics/` - Performance visualizations and reports
- `app/predict.py` - Disease prediction logic
- `utils/sensor_simulator.py` - Health monitoring simulator

## 🛠️ Technologies

- Flask, Python 3.11
- TensorFlow, Keras
- SQLite, SQLAlchemy
- Bootstrap 5
- Matplotlib, Seaborn

## 📊 Features

✅ AI Disease Detection  
✅ Real-time Health Monitoring  
✅ Overall Health Index Calculator  
✅ Prediction History  
✅ CSV/PDF Export  

## 📄 License

Academic Thesis Project © 2026
