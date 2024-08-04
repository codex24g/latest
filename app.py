import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Paths to the dataset directories
train_dir = 'train'
valid_dir = 'valid'
test_dir = 'test'

# Image dimensions
img_height, img_width = 224, 224
batch_size = 8

# Log TensorFlow and Streamlit versions
st.write(f"TensorFlow version: {tf.__version__}")
st.write(f"Streamlit version: {st.__version__}")

# Function to update and overwrite class_names.json with current classes
def update_and_overwrite_class_names():
    try:
        all_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        class_names = {cls: i for i, cls in enumerate(all_classes)}
        with open('class_names.json', 'w') as f:
            json.dump(class_names, f, indent=4)
        st.write("Updated class_names.json")
    except Exception as e:
        st.error(f"Error updating class_names.json: {e}")

# Function to create necessary folders
def create_folders(class_name):
    try:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        st.write(f"Created folders for class: {class_name}")
    except Exception as e:
        st.error(f"Error creating folders for class {class_name}: {e}")

# Function to create required folders if they don't exist
def create_required_folders():
    directories = [train_dir, valid_dir, test_dir]
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                st.write(f"Created directory: {directory}")
            except Exception as e:
                st.error(f"Error creating directory {directory}: {e}")

    all_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    for class_name in all_classes:
        create_folders(class_name)

# Data augmentation
def augment_image(image, count=30):
    try:
        datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        augmented_images = []

        for _ in range(count):
            augmented_image = datagen.flow(image, batch_size=1)[0]
            augmented_image = array_to_img(augmented_image[0])
            augmented_images.append(augmented_image)
        
        st.write(f"Augmented {count} images.")
        return augmented_images
    except Exception as e:
        st.error(f"Error during image augmentation: {e}")

# Save images to respective folders
def save_images(original_image, images, class_name):
    try:
        train_folder = os.path.join(train_dir, class_name)
        valid_folder = os.path.join(valid_dir, class_name)
        test_folder = os.path.join(test_dir, class_name)
        
        original_image.save(os.path.join(test_folder, 'original_image.jpg'))
        
        for i, img in enumerate(images):
            if i < 3:
                img.save(os.path.join(test_folder, f'image_{i}.jpg'))
            elif i < 9:
                img.save(os.path.join(valid_folder, f'image_{i}.jpg'))
            else:
                img.save(os.path.join(train_folder, f'image_{i}.jpg'))
        
        st.write("Images saved successfully.")
    except Exception as e:
        st.error(f"Error saving images: {e}")

# Preprocess image for prediction
def preprocess_image(image):
    try:
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")

# Retrain the model
def retrain_model():
    try:
        update_and_overwrite_class_names()

        with open('class_names.json', 'r') as f:
            class_names = json.load(f)

        num_classes = len(class_names)
        st.write(f"Number of classes: {num_classes}")

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        valid_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical'
        )

        valid_generator = valid_datagen.flow_from_directory(
            valid_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical'
        )

        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        st.write("Starting model training...")
        history = model.fit(
            train_generator,
            epochs=5,
            validation_data=valid_generator
        )

        # Log training results
        st.write(f"Training loss: {history.history['loss'][-1]}")
        st.write(f"Validation accuracy: {history.history['val_accuracy'][-1]}")

        # Evaluate the model on the test set
        st.write("Evaluating model on test set...")
        test_loss, test_accuracy = model.evaluate(test_generator)
        st.write(f"Test Loss: {test_loss:.4f}")
        st.write(f"Test Accuracy: {test_accuracy:.4f}")

        model.save('staff_mobilenet_v2_model.h5')
        st.write("Model saved successfully.")
    except Exception as e:
        st.error(f"Error during model retraining: {e}")

# Streamlit app layout
st.title("Staff Image Recognition")

st.write("Upload an image of a staff member to get predictions and add new images to the dataset.")

class_name = st.text_input("Enter the class name for the new images:")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

create_required_folders()

if uploaded_image is not None and class_name:
    create_folders(class_name)

    try:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        augmented_images = augment_image(image)
        save_images(image, augmented_images, class_name)

        retrain_model()

        try:
            model = tf.keras.models.load_model('staff_mobilenet_v2_model.h5')
            st.write("Model loaded successfully.")
        except Exception as e:
            st.error(f"Error loading model: {e}")

        processed_image = preprocess_image(image)
        st.write(f"Processed image shape: {processed_image.shape}")
        st.write(f"Processed image stats: min={np.min(processed_image)}, max={np.max(processed_image)}")

        predictions = model.predict(processed_image)
        st.write(f"Prediction probabilities: {predictions}")

        predicted_class_index = np.argmax(predictions)
        st.write(f"Predicted class index: {predicted_class_index}")

        with open('class_names.json', 'r') as f:
            class_names = json.load(f)

        st.write(f"Class names loaded from class_names.json: {class_names}")

        predicted_class = list(class_names.keys())[predicted_class_index]
        st.write(f"Prediction: {predicted_class} (Class Index: {predicted_class_index}) with probability {np.max(predictions):.2f}")

    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")
else:
    if uploaded_image is None:
        st.write("Please upload an image.")
    if not class_name:
        st.write("Please enter a class name.")
