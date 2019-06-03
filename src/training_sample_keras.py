from autcar import Trainer
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization, InputLayer, Activation

input_folder_path = "src/ml/data/autcar_training"
output_folder_path = "src/ml/data/autcar_training_balanced"
image_width = 224
image_height = 168

trainer = Trainer(deeplearning_framework="keras", image_height=image_height, image_width=image_width)
trainer.create_balanced_dataset(input_folder_path, output_folder_path=output_folder_path)

model = Sequential([
    InputLayer(input_shape=[3,image_height,image_width]),
    Conv2D(filters=32, kernel_size=5, strides=1, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=8, padding='same'),
    Conv2D(filters=48, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=5, padding='same'),
    Conv2D(filters=64, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=3, padding='same'),
    Conv2D(filters=32, kernel_size=5, strides=1, padding='same'),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.1),
    Dense(12, activation='softmax')
])

trainer.train(output_folder_path, model, epochs=5, output_model_path="driver_keras.onnx")
trainer.test("driver_keras.onnx", output_folder_path+"/test_map.txt")