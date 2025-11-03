# Simple test code only
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the model
model = tf.keras.models.load_model('D:\point\day_night_cnn_model.h5')
print("✅ Model loaded successfully")

# 2. Define prediction function
def predict_image(model, image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not load the image!")
        return None
    
    # Process the image
    img = cv2.resize(img, (100, 100))  # Changed from 150 to 100 to match training
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Prediction
    prediction = model.predict(img)
    classes = ['Day ☀️', 'Night 🌙']
    result = classes[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    # Display result
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title(f'Result: {result} (Confidence: {confidence:.2%})')
    plt.axis('off')
    plt.show()
    
    return result, confidence

# 3. Testing
image_path = 'D:\point\k11.jpg'
result = predict_image(model, image_path)
print(f"🎯 Result: {result}")


#  طريقة اسكرين windows + shift + s