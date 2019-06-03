from autcar import Trainer
from cntk.layers import Dense, Sequential, Activation, Convolution2D, MaxPooling, Dropout, BatchNormalization
from cntk import softmax, relu

input_folder_path = "src/ml/data/autcar_training"
output_folder_path = "src/ml/data/autcar_training_balanced"
image_width = 224
image_height = 168

trainer = Trainer(deeplearning_framework="cntk", image_height=image_height, image_width=image_width)
#trainer.create_balanced_dataset(input_folder_path, output_folder_path=output_folder_path)
num_classes = trainer.get_no_classes(output_folder_path)

model = Sequential([
    Convolution2D(filter_shape=(5,5), num_filters=32, strides=(1,1), pad=True, name="first_conv"),
    BatchNormalization(map_rank=1),
    Activation(relu),
    MaxPooling(filter_shape=(3,3), strides=(2,2), name="first_max"),
    Convolution2D(filter_shape=(3,3), num_filters=48, strides=(1,1), pad=True, name="second_conv"),
    BatchNormalization(map_rank=1),
    Activation(relu),
    MaxPooling(filter_shape=(3,3), strides=(2,2), name="second_max"),
    Convolution2D(filter_shape=(3,3), num_filters=64, strides=(1,1), pad=True, name="third_conv"),
    BatchNormalization(map_rank=1),
    Activation(relu),
    MaxPooling(filter_shape=(3,3), strides=(2,2), name="third_max"),
    Convolution2D(filter_shape=(5,5), num_filters=32, strides=(1,1), pad=True, name="fourth_conv"),
    BatchNormalization(map_rank=1),
    Activation(relu),
    Dense(100, activation=relu),
    Dropout(0.1),
    Dense(12, activation=softmax)
])

# trainer.train(output_folder_path, model, epochs=4, output_model_path="driver_cntk.onnx")
trainer.test("driver_cntk.onnx", output_folder_path+"/test_map.txt")