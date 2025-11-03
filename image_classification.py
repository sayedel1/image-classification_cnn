import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf

def load_images_from_folders():
    images = []
    labels = []
    
    # Day folder
    day_folder = "dataset/day"
    if os.path.exists(day_folder):
        for filename in os.listdir(day_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(day_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (100, 100))  
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                    images.append(img)
                    labels.append(0)  # 0 for day
    
    # Night folder
    night_folder = "dataset/night" 
    if os.path.exists(night_folder):
        for filename in os.listdir(night_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(night_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (100, 100))  
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                    images.append(img)
                    labels.append(1)  # 1 for night
    
    return np.array(images), np.array(labels)

def build_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # Day and night
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def main():
    print("📁 Loading images...")
    
    # Load images
    X, y = load_images_from_folders()
    
    if len(X) == 0:
        print("❌ No images found!")
        print("⚠️ Please check:")
        print("   - 'dataset/day' folder exists with day images")
        print("   - 'dataset/night' folder exists with night images")
        return
    
    print(f"✅ Successfully loaded {len(X)} images")
    print(f"   - {np.sum(y == 0)} day images")
    print(f"   - {np.sum(y == 1)} night images")
    
    # Normalize data
    X = X.astype('float32') / 255.0
    
    # Convert labels to one-hot encoding
    y_categorical = to_categorical(y, 2)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42
    )
    
    print(f"📊 Data split:")
    print(f"   - Training data: {len(X_train)} images")
    print(f"   - Test data: {len(X_test)} images")
    
    # Build model
    print("🧠 Building model...")
    model = build_cnn_model(X_train[0].shape)
    model.summary()
    
    # Training
    print("🚀 Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate model
    print("📈 Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Model accuracy on test data: {test_accuracy:.2%}")
    
    # Plot training results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Save model
    model.save('day_night_cnn_model.h5')
    print("💾 Model saved as 'day_night_cnn_model.h5'")
    
    return model

# Function to predict a new image
def predict_image(model, image_path):
    """
    Predict classification of a new image (day or night)
    """
    if not os.path.exists(image_path):
        print("❌ Image not found!")
        return None
    
    # Load and prepare image
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error loading image!")
        return None
    
    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  
    
    # Prediction
    prediction = model.predict(img)
    class_names = ['Day ☀️', 'Night 🌙']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    # Display result
    print(f"🎯 Classification: {predicted_class}")
    print(f"📊 Confidence level: {confidence:.2%}")
    
    # Display image
    img_display = cv2.imread(image_path)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img_display)
    plt.title(f'Result: {predicted_class}\nConfidence: {confidence:.2%}')
    plt.axis('off')
    plt.show()
    
    return predicted_class, confidence


def create_sample_images():
    """
    Create sample images if no real data available
    """
    import numpy as np
    
    os.makedirs('dataset/day', exist_ok=True)
    os.makedirs('dataset/night', exist_ok=True)
    
    print("🛠️ Creating sample images...")
    
    
    for i in range(30):
        
        img = np.random.randint(150, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(f'dataset/day/day_{i+1}.jpg', img)
    
    
    for i in range(30):
        
        img = np.random.randint(0, 100, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(f'dataset/night/night_{i+1}.jpg', img)
    
    print("✅ Created 60 sample images (30 day, 30 night)")


if __name__ == "__main__":
    print("=" * 50)
    print("   Image Classification - CNN with OpenCV")
    print("=" * 50)
    
    
    if not os.path.exists('dataset/day') or not os.path.exists('dataset/night'):
        print("⚠️ No data found, creating sample data...")
        create_sample_images()
    
    
    model = main()
    
    if model:
        print("\n" + "=" * 40)
        print("   Model Testing")
        print("=" * 40)
        
        
        print("To test the model on a new image, use:")
        print("predict_image(model, 'your_image_path.jpg')")
        
        # Example if you have a test image:
        # test_image = "test.jpg"
        # if os.path.exists(test_image):
        #     predict_image(model, test_image)