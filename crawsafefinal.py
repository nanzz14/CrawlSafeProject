import tensorflow as tf
import os
import numpy as np
import cv2
import skimage
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import gc
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st
from keras.models import load_model
from PIL import Image
from collections import Counter


batch_size = 64
imageSize = 64
train_len = 631  # Update this based on actual number of images in your dataset
train_dir ='C:\\Users\\ig134\\projects\\helloworld\\crawlsafe\\train' # Path to your training dataset

# Function to load and preprocess images
def get_data(folder):
    X = np.empty((train_len, imageSize, imageSize, 3), dtype=np.float32)
    y = np.empty((train_len,), dtype=int)
    cnt = 0
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName == 'cockroaches':
                label = 0  # Cockroach class
            else:
                label = 1  # Other (Nothing Detected)

            for image_filename in os.listdir(os.path.join(folder, folderName)):
                img_file = cv2.imread(os.path.join(folder, folderName, image_filename))
                if img_file is not None:
                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
                    X[cnt] = img_file
                    y[cnt] = label
                    cnt += 1
    return X[:cnt], y[:cnt]  # Trim arrays to actual size

# Load the training data
X_train, y_train = get_data(train_dir)
print("Images imported..")
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")

# Check class distribution
class_distribution = Counter(y_train)
print("Class distribution:", class_distribution)

# Ensure there are at least 2 samples in each class
if min(class_distribution.values()) < 2:
    print("Warning: Some classes have fewer than 2 samples. Stratified split may not work.")
    # Option 1: Do not use stratification
    stratify_option = None
else:
    stratify_option = y_train

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42, stratify=stratify_option
)

# One-hot encoding the labels
y_cat_train = to_categorical(y_train, 2)
y_cat_test = to_categorical(y_test, 2)

# Cleaning up
gc.collect()

# Model Architecture
model = Sequential()

# First Conv Layer
model.add(Conv2D(32, (5, 5), input_shape=(64, 64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Conv Layer
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third Conv Layer
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten and Fully Connected Layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))

# Output Layer for 2 classes (Cockroach or Nothing Detected)
model.add(Dense(2, activation='softmax'))

# Model Summary
model.summary()

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_cat_train,
                    validation_data=(X_test, y_cat_test),
                    epochs=10, batch_size=batch_size,
                    callbacks=[early_stopping])

# Evaluate the model
model.evaluate(X_test, y_cat_test, verbose=0)

# Save the model
model.save('crawl_safe.keras')
print("Model saved.")

# Define the uploads directory
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the uploads directory if it doesn't exist

# Load the saved model (make sure the model is present in the same directory or provide the full path)
model = load_model('crawl_safe.keras')

def check_food_safety(file_path):
    """
    This function loads the image file, processes it, and uses the pre-trained model to detect pests.
    """
    # Read the uploaded image
    img = cv2.imread(file_path)

    if img is None:
        return "Invalid image file."

    # Preprocess the image
    resized_img = cv2.resize(img, (64, 64))  # Resize the image to match the input shape
    input_img = resized_img.reshape(1, 64, 64, 3)  # Reshape to match model input
    input_img = input_img / 255.0  # Normalize the image

    # Get predictions
    predictions = model.predict(input_img)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Check if the predicted class is cockroach (assuming 0 is the class label for cockroach)
    if predicted_class == 0:
        label = "Cockroach Detected"
    else:
        label = "Nothing detected"

    # Annotate the original image with the result
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Save the image with annotation to display it in Streamlit
    result_img_path = os.path.join(UPLOAD_FOLDER, "result_" + os.path.basename(file_path))
    cv2.imwrite(result_img_path, img)

    return label, result_img_path

# Streamlit app layout
st.title("Detect Pest")
st.write("Upload a file to check for pests.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a file...", type=["jpg", "png", "jpeg", "mp4"])

if uploaded_file is not None:
    # Save the uploaded file to the uploads folder
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())  # Save the file

    # Get file size
    file_size = os.path.getsize(file_path)  # Get the file size in bytes

    # Call the check_food_safety function with the file path as an argument
    safety_message, result_img_path = check_food_safety(file_path)

    # Display the results
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")
    st.write(f"File Size: {file_size} bytes")
    st.write(f"Safety Message: {safety_message}")

    # Display the processed image with annotation
    st.image(result_img_path, caption="Processed Image", use_column_width=True)