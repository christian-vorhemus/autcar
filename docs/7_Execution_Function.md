

# Run multiple models on the car with a custom execution function

The goal of this tutorial is to execute two models on the car, one following the track and one recognizing traffic signs. The car should behave accordingly.

## Prerequisites

Make sure you have two models prepared you want to test. For this tutorial, we use a model that follows a track as described [here](3_Autonomous_Driving.md) and another model which recognizes traffic signs as described [here](5_Customvision.md). Our second model uses a custom image preprocessor, we'll reuse this here. All in all, this is the code we'll start with:

```python
from autcar import Car, Driver, Camera, Model
import numpy as np
import cv2

car = Car()
cam = Camera(rotation=-1)

class OwnModel(Model):
  def preprocess(self, image):
    image = image.convert('RGB')
    r,g,b = np.array(image).T
    image = np.array([b,g,r]).transpose()
    h, w = image.shape[:2]
    min_dim = min(w,h)
    startx = w//2-(min_dim//2)
    starty = h//2-(min_dim//2)
    image = image[starty:starty+min_dim, startx:startx+min_dim]
    resized_image = np.array([cv2.resize(image, (224, 224), interpolation = cv2.INTER_LINEAR)])
    
    return resized_image
    
model_drive = Model("driver_keras.onnx", execution_interval=1.5)
model_traffic_sign = OwnModel("trafficsign.onnx", execution_interval=3)

```

We have two model objects `model_drive` and `model_traffic_sign` ready to work with. Let's define the logic the car should follow: In general it should just follow the track, this is the job of _model_drive_. When a stop sign is visible, the prediction of _model_traffic_sign_ should overrule the predictions of _model_drive_ - the car should stop for a few seconds. Then, the car should continue driving but since it hasn't moved, it will recognize the stop sign again and stop. To prevent this, we ignore every sign recognition for a few iterations. When a priorty road sign is recognized, the car should drive faster for a few seconds and get back to normal after a few seconds.

Let's try to bring the logic above into a Python function.

When you have trained your `model_drive` model with `train()` of "AutTrainer", the model has 12 possible outputs. This corresponds to the 12 possible ways the car can move. Sorted alphabetically, here are the 12 movements and the corresponding class labels:

<img src="../images/controls.png" width="800">

So when `model_drive` outputs "0" it means drive a little bit to the left backwards. If it outputs "4" it means move fast forward and so on. In our training, it is unlikely that we use all commands, therefore a lot of labels will not be used.

<img src="../images/execution_function.png" width="500">
