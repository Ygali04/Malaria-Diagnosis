import streamlit as st
from joblib import load
import numpy as np
import cv2

model = load("model.joblib")

st.title('Malaria Detection') #Change the title here!

uploaded_file = st.file_uploader("Upload File")
if uploaded_file is not None:    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption='Image Caption')
    small = cv2.resize(image,(50,50))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    flatten = np.reshape(gray, (1,2500))
    result = model.predict(flatten)
    if result[0] == 1:
      answer = 'Yes T_T'
    else:
      answer = 'No! :D'
    st.write("Malaria?:", answer)
else:
    pass
  
public_url = ngrok.connect(port='80')
print (public_url)
!streamlit run --server.port 80 app.py >/dev/null
