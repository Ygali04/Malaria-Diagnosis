uploaded_file = st.file_uploader("Upload File")
if uploaded_file is not None:    
    st.image(image, caption='Image Caption')
else:
    pass

file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
image = cv2.imdecode(file_bytes, 1)

small = cv2.resize(image,(50,50))
gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

flatten = np.reshape(gray, (1,2500))
result = model.predict(flatten)

st.write("Malaria? (1 for yes, 0 for no):", result)
