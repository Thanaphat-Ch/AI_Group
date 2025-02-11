import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
#pip install streamlit scikit-learn numpy pandas joblib

# โหลดข้อมูลตัวอย่างสำหรับการฝึกโมเดล
data = {
    'temperature': [27,27,27,33,33,30,30,30,30,27,27,30,30,27,25,25,25,30],
    'soil_type': [1,3,4,5,3,2,3,5,4,5,4,4,6,1,1,3,5,5],   
    #1=ดินร่วน, 2=ดินเหนียว, 3=ดินร่วนเหนียว, 4=ดินทราย, 5=ดินร่่วนทราย, 6=ดินร่วนเหนียวปนทราย
    'rainfall': [800,800,800,1500,1500,1500,1500,1000,1000,1300,1300,800,800,1200,2000,2000,2000,1300],
    'soil_moisture': [75,75,75,75,75,85,85,75,75,65,65,65,65,75,60,60,60,65],
    'relative_humidity' : [65,65,65,75,75,80,80,65,65,75,75,65,65,85,75,75,75,65],
    'crop_type': [1,1,1,2,2,3,3,4,4,5,5,6,6,7,8,8,8,9]  
    #1=มันสำปะหลัง, 2=อ้อย, 3=ข้าว, 4=ข้าวโพด, 5=มะม่วง, 6=ถั่วเขียว, 7=ชา, 8=มะพร้าวน้ำหอม, 9=หอมแดง
}

#แบ่งข้อมูล
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


st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🌱"
)
st.title("🌱 ระบบคาดการณ์พืชที่เหมาะสม")
st.subheader("กรอกค่าตัวแปรด้านล่างเพื่อทำนายพืชที่เหมาะสม")

soil_type_options = {
    "ดินร่วน": 1,
    "ดินเหนียว": 2,
    "ดินร่วนเหนียว": 3,
    "ดินทราย": 4,
    "ดินร่วนทราย": 5,
    "ดินร่วนเหนียวปนทราย": 6
}
<<<<<<< HEAD
=======

>>>>>>> 2325f588d12808762df874ae4c8e843ea4cf0326
col1, col2 = st.columns(2)
with col1:
    temperature = st.number_input("🌡️ อุณหภูมิ (°C)", min_value=10, max_value=50, value=30)
    soil_type = soil_type_options[st.selectbox("🧱 ประเภทดิน", options=list(soil_type_options.keys()))]
<<<<<<< HEAD
    rainfall = st.select_slider("🌧️ ปริมาณฝน (mm)", options=["800","900","1000","1100","1200","1300",
                                    "1400","1500","1600","1700","1800","1900","2000"])
=======
    rainfall = st.select_slider("🌧️ ปริมาณฝน (mm)", options=["800","900","1000","1100","1200","1300","1400","1500","1600","1700","1800","1900","2000"])
>>>>>>> 2325f588d12808762df874ae4c8e843ea4cf0326
with col2:
    soil_moisture = st.number_input("💧 ความชื้นในดิน (%)", min_value=10, max_value=100, value=65)
    relative_humidity = st.number_input("💦 ความชื้นสัมพัทธ์ (%)", min_value=10, max_value=100, value=70)

<<<<<<< HEAD
int(rainfall)

=======
>>>>>>> 2325f588d12808762df874ae4c8e843ea4cf0326
if st.button("🔍 ทำนายพืชที่เหมาะสม"):
    user_input = np.array([[temperature, soil_type, rainfall, soil_moisture, relative_humidity]])
    prediction = model.predict(user_input)[0]
    crop_dict = {1: "🥔 มันสำปะหลัง", 2: "🌾 อ้อย", 3: "🌾 ข้าว", 4: "🌽 ข้าวโพด", 5: "🥭 มะม่วง", 
                 6: "🌱 ถั่วเขียว", 7: "🍵 ชา", 8: "🥥 มะพร้าวน้ำหอม", 9: "🧅 หอมแดง"}
    crop_description_dict = {
        1: "การปลูกมันสำปะหลังเริ่มจากการเตรียมดินให้เหมาะสม ด้วยการไถดินลึกประมาณ 30-40 ซม. แล้วตากดินไว้สัก 7-14 วัน จากนั้นไถพรวนดินให้ละเอียดและปรับพื้นที่ให้เรียบ ควรใส่ปุ๋ยคอกหรือปุ๋ยหมักประมาณ 500-1,000 กก. ต่อไร่เพื่อเพิ่มธาตุอาหารในดิน    \nการเลือกพันธุ์ที่ดี เช่น พันธุ์ระยอง 72 หรือระยอง 90 ซึ่งทนโรคและให้ผลผลิตดี จากนั้นเตรียมท่อนพันธุ์ที่มีอายุประมาณ 8-12 เดือน ตัดให้มีความยาวประมาณ 20-25 ซม. และแช่ในสารป้องกันเชื้อรา ก่อนนำไปปักลงดินในระยะห่าง 1x1 เมตร หรือ 1x0.8 เมตร โดยปักท่อนพันธุ์เฉียงประมาณ 45 องศาหรือแนวตั้ง    \nการดูแลมันสำปะหลังนั้น ต้องกำจัดวัชพืชในช่วง 1-3 เดือนแรก ใส่ปุ๋ยในช่วงแรกด้วยสูตร 15-15-15 และช่วงที่ 3-4 เดือนให้ปุ๋ยสูตร 13-13-21 หรือ 0-0-60 รวมถึงการให้น้ำในช่วงแล้งและการป้องกันโรคและแมลง เช่น เพลี้ยแป้งและโรคใบด่าง    \nการเก็บเกี่ยวมันสำปะหลังจะทำเมื่ออายุประมาณ 8-12 เดือน โดยใช้จอบหรือเครื่องมือในการถอนหัวมัน หากดูแลดี ผลผลิตจะสูงอยู่ที่ 3-5 ตันต่อไร่หรือมากกว่านั้น ขึ้นอยู่กับสภาพแวดล้อมและการจัดการแปลงปลูก.",
<<<<<<< HEAD
        2: "การปลูกอ้อยเริ่มต้นจากการเตรียมดิน โดยการไถดินลึกประมาณ 20-30 ซม. แล้วพรวนดินให้ละเอียดเพื่อให้อ้อยสามารถเจริญเติบโตได้ดี การเตรียมดินควรทำการใส่ปุ๋ยคอกหรือปุ๋ยหมักเพื่อเพิ่มธาตุอาหารในดิน รวมถึงการปรับสภาพดินให้มีความร่วนซุยและระบายน้ำได้ดี	\nสำหรับการเลือกพันธุ์อ้อย ควรเลือกพันธุ์ที่เหมาะสมกับสภาพพื้นที่ เช่น พันธุ์อ้อยบี 9 หรือพันธุ์อ้อยอี 4 ซึ่งมีความทนทานต่อโรคและให้ผลผลิตสูง	\nการปลูกอ้อยจะทำโดยการตัดท่อนพันธุ์อ้อยที่มีความยาวประมาณ 30-40 ซม. และปักลงในดินโดยให้ท่อนพันธุ์วางขนานไปกับดินหรือให้ส่วนปลายหันขึ้นเล็กน้อย โดยมีระยะปลูกประมาณ 1-1.5 เมตร ขึ้นอยู่กับชนิดพันธุ์และลักษณะของพื้นที่	\nการดูแลอ้อยหลังการปลูกจำเป็นต้องมีการให้น้ำในช่วงแล้ง การใส่ปุ๋ยในช่วงต่าง ๆ เช่น ปุ๋ยไนโตรเจนในช่วงเริ่มเจริญเติบโต และปุ๋ยโพแทสเซียมในช่วงอ้อยเริ่มสร้างน้ำตาล การกำจัดวัชพืชในช่วงต้น ๆ รวมถึงการป้องกันและรักษาโรคที่อาจเกิดขึ้น เช่น โรคใบไหม้ โรคเหี่ยว และแมลงศัตรูพืช	\nการเก็บเกี่ยวอ้อยจะทำเมื่ออ้อยมีอายุประมาณ 12-18 เดือน หรือเมื่อได้ขนาดที่ต้องการ โดยการตัดลำอ้อยด้วยมือหรือใช้เครื่องจักรในการตัด เพื่อให้ได้อ้อยที่มีน้ำตาลสูงและพร้อมสำหรับการใช้งานต่าง ๆ เช่น ผลิตน้ำตาล หรือเป็นอาหารสัตว์.",
=======
        2: "ใการปลูกอ้อยเริ่มต้นจากการเตรียมดิน โดยการไถดินลึกประมาณ 20-30 ซม. แล้วพรวนดินให้ละเอียดเพื่อให้อ้อยสามารถเจริญเติบโตได้ดี การเตรียมดินควรทำการใส่ปุ๋ยคอกหรือปุ๋ยหมักเพื่อเพิ่มธาตุอาหารในดิน รวมถึงการปรับสภาพดินให้มีความร่วนซุยและระบายน้ำได้ดี	\nสำหรับการเลือกพันธุ์อ้อย ควรเลือกพันธุ์ที่เหมาะสมกับสภาพพื้นที่ เช่น พันธุ์อ้อยบี 9 หรือพันธุ์อ้อยอี 4 ซึ่งมีความทนทานต่อโรคและให้ผลผลิตสูง	\nการปลูกอ้อยจะทำโดยการตัดท่อนพันธุ์อ้อยที่มีความยาวประมาณ 30-40 ซม. และปักลงในดินโดยให้ท่อนพันธุ์วางขนานไปกับดินหรือให้ส่วนปลายหันขึ้นเล็กน้อย โดยมีระยะปลูกประมาณ 1-1.5 เมตร ขึ้นอยู่กับชนิดพันธุ์และลักษณะของพื้นที่	\nการดูแลอ้อยหลังการปลูกจำเป็นต้องมีการให้น้ำในช่วงแล้ง การใส่ปุ๋ยในช่วงต่าง ๆ เช่น ปุ๋ยไนโตรเจนในช่วงเริ่มเจริญเติบโต และปุ๋ยโพแทสเซียมในช่วงอ้อยเริ่มสร้างน้ำตาล การกำจัดวัชพืชในช่วงต้น ๆ รวมถึงการป้องกันและรักษาโรคที่อาจเกิดขึ้น เช่น โรคใบไหม้ โรคเหี่ยว และแมลงศัตรูพืช	\nการเก็บเกี่ยวอ้อยจะทำเมื่ออ้อยมีอายุประมาณ 12-18 เดือน หรือเมื่อได้ขนาดที่ต้องการ โดยการตัดลำอ้อยด้วยมือหรือใช้เครื่องจักรในการตัด เพื่อให้ได้อ้อยที่มีน้ำตาลสูงและพร้อมสำหรับการใช้งานต่าง ๆ เช่น ผลิตน้ำตาล หรือเป็นอาหารสัตว์.",
>>>>>>> 2325f588d12808762df874ae4c8e843ea4cf0326
        3: "การปลูกข้าวเริ่มต้นจากการเตรียมดิน โดยการไถดินและพรวนดินให้ละเอียดเพื่อลดการอุดตันของดินและทำให้รากข้าวสามารถเจริญเติบโตได้ดี ถ้าปลูกในแปลงนาก็ควรทำการปรับสภาพดินให้มีความชุ่มชื้นและมีน้ำขังพอเหมาะ เพราะข้าวเป็นพืชที่ชอบน้ำ	\nหลังจากเตรียมดินแล้ว ควรเลือกพันธุ์ข้าวที่เหมาะสมกับสภาพพื้นที่ เช่น ข้าวหอมมะลิ ข้าวเหนียว หรือข้าวปทุมธานี ซึ่งแต่ละพันธุ์จะมีคุณสมบัติที่แตกต่างกันไปในเรื่องการทนทานต่อโรคและการให้ผลผลิต	\nการเพาะเมล็ดข้าวจะทำโดยการแช่เมล็ดข้าวในน้ำประมาณ 1-2 วัน ก่อนที่จะนำไปเพาะในแปลงเพาะเมล็ด หรือหากใช้การปลูกโดยตรงสามารถใช้เมล็ดข้าวที่ปลูกแล้วโรยลงในแปลงได้เลย เมื่อพืชข้าวเจริญเติบโตจนมีใบอ่อนและตั้งต้นแล้ว จึงย้ายไปปลูกในแปลงนา	\nการปลูกข้าวในแปลงนาจะทำโดยการปลูกในระยะห่างประมาณ 25-30 ซม. ซึ่งเหมาะสมกับพื้นที่และการเติบโตของข้าว หากเป็นการปลูกในนา ต้องให้แปลงนาเต็มไปด้วยน้ำในช่วงต้นเพื่อให้ต้นข้าวสามารถเติบโตได้ดี โดยปกติจะต้องมีน้ำขังในแปลงนาอย่างน้อย 5-7 ซม. ตั้งแต่ช่วงต้นจนถึงช่วงออกรวง	\nการดูแลข้าวในระหว่างการปลูก ได้แก่ การกำจัดวัชพืช ซึ่งอาจใช้วิธีการพรวนดินหรือใช้สารเคมีในการกำจัดวัชพืช รวมทั้งการใส่ปุ๋ย โดยปุ๋ยที่ใช้จะมีการใส่ปุ๋ยในช่วงเริ่มปลูกและช่วงที่ข้าวเจริญเติบโตแล้ว ซึ่งปุ๋ยไนโตรเจนจะช่วยให้ข้าวเติบโตได้ดีและมีใบที่แข็งแรง	\nการป้องกันโรคและแมลง เช่น โรคขอบใบแห้ง หรือแมลงเช่น เพลี้ยไฟ เป็นสิ่งที่ต้องดูแลตลอดเวลา ในช่วงที่ข้าวเริ่มออกดอกและออกรวง ควรป้องกันการเข้าทำลายของโรคต่าง ๆ ด้วยการใช้สารเคมีหรือวิธีการธรรมชาติ	\nการเก็บเกี่ยวข้าวจะทำเมื่อข้าวมีอายุประมาณ 3-4 เดือน หรือเมื่อเมล็ดข้าวเริ่มสุกและมีสีทอง การเก็บเกี่ยวควรทำในช่วงที่อากาศแห้ง เพื่อป้องกันไม่ให้ข้าวเน่าเสียระหว่างการเก็บรวบรวม จากนั้นนำข้าวไปตากให้แห้งเพื่อเตรียมการเก็บรักษาหรือการขายต่อไป.",
        4: "การปลูกข้าวโพดเริ่มจากการเตรียมดิน โดยการไถดินและพรวนดินให้ละเอียดเพื่อให้อากาศและน้ำสามารถเข้าสู่รากได้ดี ข้าวโพดเป็นพืชที่ต้องการดินที่ร่วนซุยและระบายน้ำดี นอกจากนี้ ควรใส่ปุ๋ยคอกหรือปุ๋ยหมักก่อนปลูกเพื่อเพิ่มธาตุอาหารในดิน	\nเมื่อเตรียมดินเสร็จแล้ว ควรเลือกพันธุ์ข้าวโพดที่เหมาะสมกับสภาพพื้นที่ เช่น ข้าวโพดหวาน ข้าวโพดเลี้ยงสัตว์ หรือข้าวโพดพันธุ์ลูกผสมที่ให้ผลผลิตสูง หลังจากนั้นให้เตรียมเมล็ดข้าวโพดสำหรับการปลูก โดยเมล็ดข้าวโพดควรมีขนาดใหญ่และสมบูรณ์	\nการปลูกข้าวโพดจะใช้วิธีการหยอดเมล็ดลงไปในแปลง โดยให้ระยะห่างระหว่างเมล็ดประมาณ 20-30 ซม. และระยะระหว่างแถวประมาณ 70-90 ซม. การปลูกข้าวโพดในแปลงควรทำในช่วงฤดูฝนหรือฤดูที่มีน้ำเพียงพอ เพราะข้าวโพดต้องการน้ำอย่างสม่ำเสมอในช่วงการเจริญเติบโต	\nการดูแลข้าวโพดประกอบด้วยการใส่ปุ๋ย โดยปุ๋ยที่ใช้จะเป็นปุ๋ยที่มีไนโตรเจนสูงในช่วงต้นเพราะช่วยให้ข้าวโพดเติบโตแข็งแรง และในช่วงที่ข้าวโพดเริ่มออกรวงจะต้องใส่ปุ๋ยที่มีฟอสฟอรัสและโพแทสเซียมเพื่อช่วยในการสร้างผลผลิต การให้ปุ๋ยควรแบ่งเป็นหลายครั้งตามระยะการเจริญเติบโตของข้าวโพด	\nการกำจัดวัชพืชและการป้องกันโรคหรือแมลงเป็นสิ่งสำคัญในการปลูกข้าวโพด เช่น เพลี้ยอ่อน หนอนเจาะลำต้น หรือโรครากเน่า ซึ่งอาจทำลายผลผลิตได้ การใช้สารเคมีหรือวิธีธรรมชาติในการป้องกันศัตรูพืชก็เป็นวิธีที่ต้องทำอย่างสม่ำเสมอ	\nการเก็บเกี่ยวข้าวโพดจะทำเมื่อผลมีอายุประมาณ 3-4 เดือน หรือเมื่อเมล็ดเริ่มสุกและแข็งเต็มที่ การเก็บเกี่ยวควรทำในช่วงที่ฝักข้าวโพดมีสีทองหรือเหลืองเต็มที่ และฝักสามารถแยกออกจากลำต้นได้ง่าย หลังจากเก็บเกี่ยวแล้ว ข้าวโพดจะถูกนำไปตากให้แห้งเพื่อเตรียมการเก็บรักษาหรือการขาย.",
        5: "การปลูกมะม่วงเริ่มต้นจากการเลือกพันธุ์ที่เหมาะสมกับสภาพพื้นที่ เช่น มะม่วงน้ำดอกไม้, มะม่วงเขียวเสวย, หรือมะม่วงมหาชนก ซึ่งแต่ละพันธุ์มีลักษณะและรสชาติที่แตกต่างกัน การเลือกพันธุ์ควรคำนึงถึงสภาพอากาศและดินในพื้นที่ปลูก	\nขั้นตอนแรกของการปลูกมะม่วงคือการเตรียมดินโดยการขุดหลุมปลูกขนาดใหญ่ประมาณ 60x60x60 ซม. และใส่ปุ๋ยคอกหรือปุ๋ยหมักลงไปในหลุมประมาณ 1-2 ถัง เพื่อเพิ่มธาตุอาหารในดิน จากนั้นควรทำการปรับระดับดินและทำให้ดินร่วนซุย	\nสำหรับการปลูกมะม่วง ควรปลูกในระยะห่างประมาณ 8-10 เมตร ระหว่างต้น เพราะมะม่วงต้องการพื้นที่ในการเจริญเติบโตและรับแสงแดดอย่างเต็มที่ การปลูกมะม่วงควรปลูกในช่วงฤดูฝนเพื่อให้ต้นกล้ามีการเจริญเติบโตดีและมีน้ำเพียงพอต่อการพัฒนา	\nการดูแลมะม่วงหลังการปลูกมีความสำคัญ โดยการรดน้ำในช่วงต้นที่ปลูกใหม่จนกว่าต้นจะเริ่มตั้งตัวได้ดี ในระยะ 2-3 ปีแรก ควรตัดแต่งกิ่งเพื่อลดความแออัดและให้ต้นเจริญเติบโตดีขึ้น รวมถึงการใส่ปุ๋ยเพื่อให้ต้นมะม่วงได้รับธาตุอาหารที่จำเป็น โดยปุ๋ยที่ใช้ควรมีทั้งไนโตรเจน ฟอสฟอรัส และโพแทสเซียม	\nการป้องกันโรคและแมลง เช่น โรคแอนแทรคโนส, เพลี้ยไฟ, หรือหนอนมะม่วง เป็นสิ่งที่ต้องทำอย่างสม่ำเสมอ การใช้สารเคมีป้องกันโรคหรือการใช้วิธีธรรมชาติในการควบคุมแมลงจะช่วยให้ต้นมะม่วงมีสุขภาพดีและผลผลิตสูง	\nการเก็บเกี่ยวมะม่วงจะทำเมื่อผลมะม่วงมีอายุประมาณ 3-5 เดือน (ขึ้นอยู่กับพันธุ์) และเริ่มมีสีเปลี่ยนจากเขียวเป็นสีเหลืองหรือสีส้ม สำหรับมะม่วงที่กินดิบจะเก็บก่อนที่จะสุกเต็มที่ ส่วนมะม่วงที่ต้องการสุกในต้นก็สามารถปล่อยให้สุกจนถึงเวลาที่เหมาะสมก่อนเก็บเกี่ยว นอกจากนี้ การเก็บมะม่วงควรทำอย่างระมัดระวังไม่ให้ผลมะม่วงบอบช้ำหรือเสียหายจากการขนส่ง.",
        6: "การปลูกถั่วเขียวเริ่มจากการเลือกพันธุ์ที่ดี เช่น ถั่วเขียวพันธุ์พื้นเมือง หรือพันธุ์ที่ปรับปรุงพันธุ์เพื่อให้ผลผลิตสูงและทนทานต่อโรค เช่น พันธุ์ถั่วเขียวพันธุ์ปรับปรุงจากกรมวิชาการเกษตร	\nขั้นตอนแรกในการปลูกถั่วเขียวคือการเตรียมดิน โดยการไถดินให้ละเอียดและพรวนดินให้ร่วนซุย เพื่อให้รากถั่วเจริญเติบโตได้ดี ถั่วเขียวชอบดินร่วนซุย ระบายน้ำดีและไม่ชอบดินที่อมน้ำ หากดินมีความเป็นกรดหรือด่างมากเกินไป ควรปรับค่า pH ของดินให้เหมาะสมที่ประมาณ 6-7	\nการปลูกถั่วเขียวจะทำโดยหยอดเมล็ดลงในดินที่เตรียมไว้ โดยให้ระยะห่างระหว่างต้นประมาณ 15-20 ซม. และระหว่างแถวประมาณ 30-40 ซม. ซึ่งจะทำให้ถั่วเขียวมีพื้นที่เติบโตดีและไม่แออัด เมล็ดถั่วควรฝังลึกประมาณ 2-3 ซม. เพื่อป้องกันเมล็ดจากแมลงหรือสัตว์ที่มาขุดกิน	\nการดูแลถั่วเขียวจะเน้นที่การรดน้ำอย่างสม่ำเสมอ แต่ไม่ให้ดินแฉะเกินไป เนื่องจากถั่วเขียวไม่ชอบน้ำขัง ในช่วงการเจริญเติบโต ควรให้ปุ๋ยที่มีไนโตรเจนในปริมาณพอเหมาะ และในช่วงที่ใกล้เก็บเกี่ยวสามารถใช้ปุ๋ยที่มีฟอสฟอรัสและโพแทสเซียมเพื่อกระตุ้นการออกฝัก	\nการกำจัดวัชพืชเป็นสิ่งที่ต้องทำในช่วงต้น ๆ เพื่อไม่ให้วัชพืชแย่งน้ำและอาหารจากต้นถั่ว การป้องกันแมลงและโรค เช่น เพลี้ยไฟหรือโรครากเน่า ก็เป็นเรื่องที่ควรใส่ใจโดยการใช้วิธีธรรมชาติหรือสารเคมีที่ปลอดภัย	\nการเก็บเกี่ยวถั่วเขียวจะทำเมื่อฝักมีอายุประมาณ 2-3 เดือน และเมล็ดถั่วเริ่มแก่และแห้ง เมื่อต้นถั่วเริ่มเหี่ยวและฝักเริ่มเปิด จะเป็นสัญญาณว่าได้เวลาเก็บเกี่ยว ซึ่งจะเก็บฝักมาพักให้แห้ง แล้วนำเมล็ดถั่วไปตากให้แห้งเพื่อลดความชื้น ก่อนนำไปเก็บรักษาหรือขาย.",
        7: "การปลูกชาเริ่มจากการเลือกพันธุ์ชาที่เหมาะสม เช่น พันธุ์ชาจีน หรือพันธุ์ชาศรีลังกา ซึ่งแต่ละพันธุ์มีลักษณะเฉพาะตัวและความเหมาะสมกับสภาพภูมิอากาศต่าง ๆ โดยชาเป็นพืชที่ต้องการสภาพแวดล้อมที่เย็นและมีฝนตกสม่ำเสมอ การเลือกสถานที่ปลูกควรเลือกพื้นที่ที่มีอากาศเย็น มีความชื้นสูง และดินต้องมีการระบายน้ำดี	\nการเตรียมดินสำหรับการปลูกชาจะเริ่มด้วยการขุดดินและพรวนให้ละเอียด เพื่อให้รากของต้นชาสามารถเจริญเติบโตได้ดี ดินที่เหมาะสมกับการปลูกชาคือดินร่วนซุย มีความเป็นกรดเล็กน้อย (pH 5.5-6.5) ควรปรับดินให้เหมาะสมก่อนการปลูก	\nสำหรับการปลูกชา สามารถใช้การเพาะกล้าในกระบะเพาะหรือนำต้นกล้าชามาปลูกในแปลงปลูก โดยให้ระยะห่างระหว่างต้นประมาณ 50-75 ซม. และระหว่างแถวประมาณ 75-100 ซม. การปลูกชาจะทำในฤดูฝนเพื่อให้ต้นชามีโอกาสเติบโตได้ดีและไม่ต้องรดน้ำมากนัก	\nการดูแลต้นชาหลังการปลูกต้องให้ความสำคัญกับการรดน้ำในช่วงแรกที่ปลูกเพื่อให้ต้นชารากแข็งแรง โดยไม่ให้ดินแฉะเกินไป การตัดแต่งกิ่งเพื่อให้ต้นชามีรูปร่างที่ดีและไม่แออัดเป็นสิ่งสำคัญ การตัดแต่งจะช่วยให้ต้นชามีการเจริญเติบโตในแนวตั้ง ซึ่งทำให้ได้รับแสงแดดอย่างเต็มที่และเพิ่มผลผลิต	\nการใส่ปุ๋ยเป็นสิ่งสำคัญในการปลูกชา โดยจะใส่ปุ๋ยที่มีไนโตรเจนในช่วงต้นฤดูกาลเพื่อช่วยให้ต้นชามีการเจริญเติบโตที่ดี และใส่ปุ๋ยที่มีฟอสฟอรัสและโพแทสเซียมในช่วงที่เริ่มเก็บใบชาเพื่อกระตุ้นการเจริญเติบโตของใบชา	\nการป้องกันโรคและแมลงก็เป็นเรื่องที่ต้องระวัง เช่น โรคใบไหม้ หรือเพลี้ยไฟ การใช้สารเคมีหรือวิธีธรรมชาติในการควบคุมศัตรูพืชช่วยป้องกันความเสียหายที่อาจเกิดขึ้นกับผลผลิต	\nการเก็บเกี่ยวใบชาจะเริ่มจากการเก็บใบที่อายุ 3-4 เดือน ซึ่งจะเริ่มมีสีเขียวเข้มและมีความสด ใบชาใหม่ที่เก็บจะมีรสชาติที่ดีที่สุด โดยสามารถเก็บได้ทั้งปี ขึ้นอยู่กับสภาพอากาศและวิธีการดูแล ต้นชาสามารถให้ผลผลิตได้ดีหลังจากปลูกประมาณ 3 ปี	\nการเก็บใบชาจะทำโดยการเด็ดใบอ่อนด้านบน โดยเลือกเฉพาะใบที่ยังไม่เปิดเต็มที่ เพราะใบเหล่านี้จะมีรสชาติที่ดีและเป็นที่ต้องการของตลาด เมื่อเก็บเกี่ยวแล้วจะต้องนำใบชาไปแปรรูปต่อไป เช่น การอบแห้ง การหมัก หรือการบดเพื่อนำไปชงชา.",
        8: "การปลูกมะพร้าวน้ำหอมเริ่มจากการเลือกพันธุ์ที่ดี เช่น มะพร้าวน้ำหอมพันธุ์ไทยที่มีผลขนาดใหญ่ น้ำหวานมากและมีรสชาติอร่อย พันธุ์นี้เหมาะกับการปลูกในพื้นที่ชายฝั่งทะเลหรือพื้นที่ที่มีอากาศร้อนชื้น	\nการเตรียมดินสำหรับการปลูกมะพร้าวน้ำหอม ต้องขุดหลุมปลูกขนาดประมาณ 60x60x60 ซม. หรือใหญ่กว่านั้นขึ้นอยู่กับสภาพดิน โดยควรปรับสภาพดินให้มีการระบายน้ำดีและไม่ขังน้ำ เพราะมะพร้าวน้ำหอมไม่ชอบน้ำขังในระยะยาว ควรใส่ปุ๋ยคอกหรือปุ๋ยหมักลงในหลุมปลูกเพื่อเพิ่มธาตุอาหารและบำรุงดิน	\nการปลูกมะพร้าวน้ำหอมจะใช้ต้นกล้าหรือเมล็ดพันธุ์ที่มีอายุประมาณ 10-12 เดือน โดยสามารถปลูกได้ทั้งในแปลงดินหรือในพื้นที่ชายทะเล แต่ต้องให้ระยะห่างระหว่างต้นประมาณ 8-10 เมตร เพื่อให้ต้นมะพร้าวมีพื้นที่ในการเติบโตและสามารถรับแสงแดดได้เต็มที่	\nการดูแลมะพร้าวน้ำหอมหลังการปลูกจะต้องการการรดน้ำให้เหมาะสม โดยเฉพาะในช่วงแรกของการปลูกจนกว่าต้นมะพร้าวจะเริ่มตั้งตัวได้ดี ควรให้ปุ๋ยในระยะที่ต้นมะพร้าวเริ่มเติบโต โดยใช้ปุ๋ยที่มีไนโตรเจนสูงในช่วงต้นเพื่อลุ้นการเจริญเติบโตที่ดีและมีใบเขียวชอุ่ม หลังจากนั้นในระยะที่ต้นมะพร้าวเริ่มเติบโตเต็มที่ ควรใช้ปุ๋ยที่มีโพแทสเซียมและฟอสฟอรัสเพื่อกระตุ้นการเจริญเติบโตของผล	\nการป้องกันโรคและแมลง เช่น โรคใบไหม้ หรือเพลี้ยแป้ง เป็นสิ่งที่ต้องระวัง โดยการใช้สารเคมีหรือวิธีธรรมชาติในการควบคุมศัตรูพืชในระหว่างการปลูก	\nการเก็บเกี่ยวมะพร้าวน้ำหอมจะทำเมื่อผลมีอายุประมาณ 9-12 เดือน โดยจะเห็นว่าผลมะพร้าวเริ่มมีน้ำหอมและสีเปลือกที่เป็นสีเขียวหรือเหลืองเล็กน้อย การเก็บมะพร้าวจะทำโดยการตัดต้นมะพร้าวจากลำต้นโดยใช้มีดตัดให้ระมัดระวังเพื่อไม่ให้ผลมะพร้าวเสียหายหรือบาดเจ็บ การเก็บผลมะพร้าวควรทำเมื่อผลมีขนาดพอเหมาะและน้ำในลูกมะพร้าวมีรสชาติหวานและหอม.",
        9: "การปลูกหอมแดงเริ่มจากการเลือกพันธุ์หอมแดงที่ดี เช่น หอมแดงพันธุ์ท้องถิ่นหรือพันธุ์ที่ปรับปรุงพันธุ์เพื่อให้ผลผลิตสูงและทนทานต่อโรค การเลือกพันธุ์ที่เหมาะสมจะช่วยให้การปลูกประสบความสำเร็จได้มากขึ้น	\nการเตรียมดินสำหรับปลูกหอมแดงต้องทำการไถดินให้ร่วนซุยและพรวนดินให้ละเอียด โดยควรปรับสภาพดินให้มีการระบายน้ำดี เพราะหอมแดงไม่ชอบน้ำขังในดิน ควรทำการใส่ปุ๋ยคอกหรือปุ๋ยหมักเพื่อเพิ่มธาตุอาหารในดินและปรับปรุงโครงสร้างดินให้ดีขึ้น	\nการปลูกหอมแดงจะใช้หัวหอมแดงที่มีอายุประมาณ 3-4 เดือน โดยหัวหอมที่เลือกใช้ควรมีขนาดพอเหมาะและไม่มีโรคหรือแมลงกัดกิน สำหรับการปลูกหอมแดง ควรปลูกในแถวที่มีระยะห่างระหว่างหัวประมาณ 10-15 ซม. และระหว่างแถวประมาณ 20-30 ซม. การปลูกหอมแดงในแถวจะช่วยให้ต้นหอมเจริญเติบโตได้ดีและมีพื้นที่สำหรับการระบายอากาศ	\nการดูแลหอมแดงหลังจากปลูกจะเน้นที่การให้ความชื้นในดินอย่างสม่ำเสมอ โดยเฉพาะในช่วงแรกของการปลูก การรดน้ำต้องระมัดระวังไม่ให้ดินแฉะเกินไป เนื่องจากหอมแดงไม่ชอบน้ำขัง ควรให้ปุ๋ยที่มีไนโตรเจนในช่วงที่ต้นหอมเริ่มเติบโต และให้ปุ๋ยที่มีฟอสฟอรัสและโพแทสเซียมในช่วงที่เริ่มออกดอกหรือก่อนเก็บเกี่ยว เพื่อกระตุ้นการเจริญเติบโตและผลผลิต	\nการป้องกันโรคและแมลง เช่น โรคเน่ารากหรือหนอนเจาะหัวหอมแดง เป็นสิ่งที่ต้องระวังในระหว่างการปลูก การใช้สารเคมีหรือวิธีธรรมชาติในการควบคุมศัตรูพืชช่วยให้การปลูกหอมแดงมีประสิทธิภาพ	\nการเก็บเกี่ยวหอมแดงจะทำเมื่อต้นหอมเริ่มแห้งและใบหอมเหลือง การเก็บเกี่ยวจะทำเมื่อหัวหอมแดงมีขนาดพอเหมาะและสุกเต็มที่ โดยจะขุดขึ้นจากดินแล้วนำไปตากแดดให้แห้งก่อนเก็บรักษาหรือขาย."
    }
<<<<<<< HEAD
    st.success(f"พืชที่เหมาะสมที่สุดคือ: {crop_dict.get(prediction)}")
    st.markdown(f"{crop_description_dict.get(prediction)}")
=======
    st.success(f"พืชที่เหมาะสมที่สุดคือ: {crop_dict.get(prediction, 'ไม่สามารถคาดการณ์ได้')}")
    crop_description = crop_description_dict.get(prediction, "ไม่มีข้อมูล")
    st.markdown(f"{crop_description}")
>>>>>>> 2325f588d12808762df874ae4c8e843ea4cf0326


  