import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
#pip install streamlit scikit-learn numpy pandas joblib

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🌱",
)


# โหลดข้อมูลตัวอย่างสำหรับการฝึกโมเดล
data = {
    'temperature': [27,27,27,33,33,30,30,30,30,27,27,30,30,27,25,25,25,30],
    'soil_type': [1,3,4,5,3,2,3,5,4,5,4,4,6,1,1,3,5,5],   #1=ดินร่วน, 2=ดินเหนียว, 3=ดินร่วนเหนียว, 4=ดินทราย, 5=ดินร่่วนทราย, 6=ดินร่วนเหนียวปนทราย
    'rainfall': [800,800,800,1500,1500,1500,1500,1000,1000,1300,1300,800,800,1200,2000,2000,2000,1300],
    'soil_moisture': [75,75,75,75,75,85,85,75,75,65,65,65,65,75,60,60,60,65],
    'relative_humidity' : [65,65,65,75,75,80,80,65,65,75,75,65,65,85,75,75,75,65],
    'crop_type': [1,1,1,2,2,3,3,4,4,5,5,6,6,7,8,8,8,9]  #1=มันสำปะหลัง, 2=อ้อย, 3=ข้าว, 4=ข้าวโพด, 5=มะม่วง, 6=ถั่วเขียว, 7=ชา, 8=มะพร้าวน้ำหอม, 9=หอมแดง
}

df = pd.DataFrame(data)
X = df.drop(columns=['crop_type'])
y = df['crop_type']

# สร้างและฝึกโมเดล
model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=18)
model.fit(X, y)

# บันทึกโมเดล
joblib.dump(model, 'crop_model.pkl')

# โหลดโมเดล
model = joblib.load('crop_model.pkl')


st.title("🌱 ระบบคาดการณ์พืชที่เหมาะสม")
st.subheader("กรอกค่าตัวแปรด้านล่างเพื่อทำนายพืชที่เหมาะสม")


col1, col2 = st.columns(2)
with col1:
    temperature = st.number_input("🌡️ อุณหภูมิ (°C)", min_value=10, max_value=50, value=30)
    soil_type = st.number_input("🧱 ประเภทดิน (1-6)", min_value=1, max_value=6, value=1)
    rainfall = st.number_input("🌧️ ปริมาณฝน (mm)", min_value=0, max_value=3000, value=1000)
with col2:
    soil_moisture = st.number_input("💧 ความชื้นในดิน (%)", min_value=10, max_value=100, value=65)
    relative_humidity = st.number_input("💦 ความชื้นสัมพัทธ์ (%)", min_value=10, max_value=100, value=70)

if st.button("🔍 ทำนายพืชที่เหมาะสม"):
    user_input = np.array([[temperature, soil_type, rainfall, soil_moisture, relative_humidity]])
    prediction = model.predict(user_input)[0]
    crop_dict = {1: "🥔 มันสำปะหลัง", 2: "🌾 อ้อย", 3: "🌾 ข้าว", 4: "🌽 ข้าวโพด", 5: "🥭 มะม่วง", 
                 6: "🌱 ถั่วเขียว", 7: "🍵 ชา", 8: "🥥 มะพร้าวน้ำหอม", 9: "🧅 หอมแดง"}
    st.success(f"พืชที่เหมาะสมที่สุดคือ: {crop_dict.get(prediction, 'ไม่สามารถคาดการณ์ได้')}")
