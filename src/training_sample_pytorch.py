from autcar import Trainer
from torch.nn import Sequential, Conv2d, BatchNorm2d, MaxPool2d, ReLU, Dropout2d, Linear, Module, LogSoftmax

input_folder_path = "src/ml/data/autcar_training"
output_folder_path = "src/ml/data/autcar_training_balanced"
image_width = 224
image_height = 168

trainer = Trainer(deeplearning_framework="pytorch", image_height=image_height, image_width=image_width)
# trainer.create_balanced_dataset(input_folder_path, output_folder_path=output_folder_path)

class DriverNet(Module):
    def __init__(self):
        super(DriverNet, self).__init__()
        self.layer1 = Sequential(
            Conv2d(3, 16, kernel_size=5, padding=1),
            BatchNorm2d(16),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = Sequential(
            Conv2d(16, 32, kernel_size=5, padding=1),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(kernel_size=5, stride=2))
        self.layer3 = Sequential(
            Conv2d(32, 32, kernel_size=5, padding=1),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(kernel_size=5, stride=2))
        self.layer4 = Sequential(
            Conv2d(32, 16, kernel_size=5, padding=1),
            BatchNorm2d(16),
            ReLU(),
            MaxPool2d(kernel_size=5, stride=2))
        self.fc = Linear(9*5*16, 12)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

model = DriverNet()
trainer.train(output_folder_path, model, epochs=5, output_model_path="driver_pytorch.onnx")
trainer.test("driver_pytorch.onnx", output_folder_path+"/test_map.txt")