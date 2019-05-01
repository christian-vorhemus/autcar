from autcar import Trainer
from cntk.layers import Dense, Sequential, Activation, Convolution2D, MaxPooling, Dropout
from cntk import softmax, relu

trainer = Trainer()
trainer.create_balanced_dataset("src/ml/data/autcar_training", outputfolder_path="src/ml/data/autcar_training_balanced")
num_classes = trainer.get_no_classes("src/ml/data/autcar_training_balanced")

model = Sequential([
    Convolution2D(filter_shape=(5,5), num_filters=32, strides=(1,1), pad=True, name="first_conv"),
    Activation(relu),
    MaxPooling(filter_shape=(3,3), strides=(2,2), name="first_max"),
    Convolution2D(filter_shape=(3,3), num_filters=48, strides=(1,1), pad=True, name="second_conv"),
    Activation(relu),
    MaxPooling(filter_shape=(3,3), strides=(2,2), name="second_max"),
    Convolution2D(filter_shape=(3,3), num_filters=64, strides=(1,1), pad=True, name="third_conv"),
    Activation(relu),
    MaxPooling(filter_shape=(3,3), strides=(2,2), name="third_max"),
    Convolution2D(filter_shape=(5,5), num_filters=32, strides=(1,1), pad=True, name="fifth_conv"),
    Activation(relu),
    Dense(100, activation=relu),
    Dropout(0.1),
    Dense(num_classes, activation=softmax)
])

trainer.train("src/ml/data/autcar_training_balanced", model, epochs=20)

trainer.test("driver_model.onnx", "src/ml/data/autcar_training_balanced/test_map.txt")