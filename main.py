import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
#pip install streamlit scikit-learn numpy pandas joblib

# โหลดข้อมูลตัวอย่างสำหรับการฝึกโมเดล
data = {
    'temperature': [25, 30, 22, 35, 28, 32, 20, 27, 33, 29],
    'humidity': [60, 55, 70, 40, 50, 45, 75, 65, 42, 58],
    'rainfall': [200, 150, 300, 100, 180, 120, 350, 250, 90, 160],
    'soil_pH': [6.5, 7.0, 5.8, 6.0, 6.3, 6.8, 5.5, 6.7, 6.1, 6.4],
    'soil_moisture': [45, 40, 50, 30, 38, 35, 55, 48, 32, 42],
    'crop_type': [1, 2, 1, 3, 2, 3, 1, 2, 3, 2]  # 1=ข้าว, 2=ข้าวโพด, 3=มันสำปะหลัง
}

df = pd.DataFrame(data)
X = df.drop(columns=['crop_type'])
y = df['crop_type']

# สร้างและฝึกโมเดล
model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
model.fit(X, y)

# บันทึกโมเดล
joblib.dump(model, 'crop_model.pkl')

# โหลดโมเดล
model = joblib.load('crop_model.pkl')

# UI ด้วย Streamlit
st.title("🌱 ระบบคาดการณ์พืชที่เหมาะสม")
st.write("กรอกค่าตัวแปรด้านล่างเพื่อทำนายพืชที่เหมาะสม")

# รับค่าจากผู้ใช้
col1, col2 = st.columns(2)
with col1:
    temperature = st.number_input("อุณหภูมิ (°C)", min_value=10, max_value=50, value=30)
    humidity = st.number_input("ความชื้น (%)", min_value=10, max_value=100, value=50)
    rainfall = st.number_input("ปริมาณฝน (mm)", min_value=0, max_value=1000, value=200)
with col2:
    soil_pH = st.number_input("ค่า pH ของดิน", min_value=4.0, max_value=9.0, value=6.5)
    soil_moisture = st.number_input("ความชื้นในดิน (%)", min_value=10, max_value=100, value=40)

if st.button("🔍 ทำนายพืชที่เหมาะสม"):
    user_input = np.array([[temperature, humidity, rainfall, soil_pH, soil_moisture]])
    prediction = model.predict(user_input)[0]
    crop_dict = {1: "🌾 ข้าว", 2: "🌽 ข้าวโพด", 3: "🥔 มันสำปะหลัง"}
    st.success(f"พืชที่เหมาะสมที่สุดคือ: {crop_dict[prediction]}")