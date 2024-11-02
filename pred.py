import numpy as np
from keras.models import load_model
from keras.preprocessing import image
# import matplotlib.pyplot as plt

# Tải mô hình
model = load_model('data/my_model.h5')

# Danh sách các lớp tương ứng với các ký hiệu
class_names = ["A","B","C","D","E","F","G","H","I","K","L",'M','N','O','P','Q','R','S','T','U','V','W','X','Y']

def predict_sign_language(img_path):
    img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
    
  
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    x = x / 255.0
    
    predictions = model.predict(x)
    predicted_class = np.argmax(predictions, axis=1)
    
    print("Predicted letter is:", class_names[predicted_class[0]])

# Ví dụ sử dụng
img_path = 'images/t.jpg'  # Thay đổi đường dẫn tới hình ảnh của bạn
predict_sign_language(img_path)
