import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# Define the dataset folder
DATASET_FOLDER = "fashion"

# File paths
TRAIN_CSV = os.path.join(DATASET_FOLDER, "fashion-mnist_train.csv")
TEST_CSV = os.path.join(DATASET_FOLDER, "fashion-mnist_test.csv")

# Load datasets
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# Convert to NumPy arrays
X_train = train_df.iloc[:, 1:].values  # All pixel values
y_train = train_df.iloc[:, 0].values   # Labels

X_test = test_df.iloc[:, 1:].values    # All pixel values
y_test = test_df.iloc[:, 0].values     # Labels

# Normalize pixel values (0-255 -> 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape for CNN input (28x28 images, 1 channel)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Display dataset shape
print(f"Train data shape: {X_train.shape}, Train labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")

# Function to visualize some sample images
def show_sample_images(images, labels, num_samples=5):
    plt.figure(figsize=(10, 4))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
  #plt.show()

# Show some sample images
show_sample_images(X_train, y_train)



# Define the CNN Model
model = Sequential([
    # Convolutional Layer 1
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),  # Reduce spatial dimensions

    # Convolutional Layer 2
    Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),  # Reduce spatial dimensions

    # Flatten Layer
    Flatten(),

    # Fully Connected (Dense) Layer
    Dense(128, activation='relu'),
    Dropout(0.5),  # Dropout for regularization

    # Output Layer (10 classes for 10 fashion categories)
    Dense(10, activation='softmax')
])

# Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Show Model Summary
model.summary()

batch_size = 64      # Number of images per batch
epochs = 5          # Number of times the model sees the full dataset

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_test, y_test))


# Plot Training & Validation Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training & Validation Loss')
#plt.show()

# Plot Training & Validation Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training & Validation Accuracy')
#plt.show()


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")


# Save the trained model
model.save('fashion_mnist_model.h5')

print("Model saved successfully!")
