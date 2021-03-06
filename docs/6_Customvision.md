
# Use Microsoft Custom Vision to train a model

In this tutorial we will learn how to use the <a href="https://www.customvision.ai/" target="_blank">customvision.ai</a> online service to train a model. This works completely without code, the trained model can be downloaded as a file. We aim to create a traffic sign detector: Our car should be capable of recognizing two different traffic signs:

<p float="left">
  <img src="../images/major_road_sign.jpg" width="200" />
  <img src="../images/stop_sign.jpg" width="200" /> 
</p>

## Why using Custom Vision?

We can train and execute our model locally without any other services. However, very often, especially when dealing with large training sets or when large computing power is required, cloud services are used. Additionally, there is a variety of services that offer a zero-code interface to train models. Custom Vision is one of them. With Custom Vision you can simply upload images, add a label and train a model. Custom Vision makes use of **transfer learning**, meaning that a base model was already trained on millions of images and the already learned model weights are fine-tuned with the images you add. This is a state-of-the art technique to create very accurate and precise models.

## Create an Azure account and configure Custom Vision

1) You need an Azure subscription to use Custom Vision. If you don't have one yet, you can start with a free account [here](https://azure.microsoft.com/free/). You must enter some information including an email address which you can then use to sign in.

2) Go to [www.customvision.ai](https://www.customvision.ai) and sign in with the email address you just provided. Accept the conditions and you should see the empty start page

  <img src="../images/customvision_1.png" width="400">

3) Click on "New Project". Add a name and create a new "Resource Group". When adding a new _Resource Group_ you have to provide a name, a subscription you want to use and the location where your resource group is placed.

4) Now you should be able to add additional configurations for your project: Choose "Classification" as a project type and "Multiclass" as the classification type. For the domain, choose a **compact** one (only compact models can be exported), let's just take "General (compact)" and for the "Export Capabilities" select "Basic platforms". Create the project.

## Create and upload training data

1) Before we can train the model, we need training data. Since it's our goal to recognize traffic signs next to our road, print out the above signs and glue them to small sticks so you can place them next to the track.

2) When your track is prepared, power up your car and start the `rc_sample.py` file in your _autcar_ folder on your Raspberry Pi. Follow the procedure as described [here](3_Autonomous_Driving.md) to learn more how to create training data. 

3) After you transferred the training data from your car to your PC, take a look at the images: We basically have three cases: Images where a stop sign is visible, images where a major road sign is visible and images where no traffic sign is visible

<p float="left">
  <img src="../images/customvision_2.png" width="150" />
  <img src="../images/customvision_3.png" width="150" />
  <img src="../images/customvision_4.png" width="150" /> 
</p>

Pick **at least 50** images for each of the three categories you want to use for training. Also make sure you roughly use the same number of images for each category.

4) In Custom Vision, click on "Add images" and choose all your selected images which contain a stop sign. 

<img src="../images/customvision_5.png" width="400" />

5) Add a label for your images. In this example, we choose "stop". Upload the images and repeat the same procedure with the remaining two categories. We named the other two labels "none" when no traffic sign is visible and "priority" for the major road sign.

<img src="../images/customvision_6.png" width="400" />

6) Click on the green "Train" button on top of your project and select "Fast Training". Training the model should take under one minute. Afterwards you'll get the training results.

## Interpret Custom Vision training results

On the "Performance" tab you should see the following diagrams after training:

<img src="../images/customvision_7.png" width="400" />

Precision tells us, for example if the model predicted that on a certain image there is a stop sign, how likely is it that there really is a stop sign visible (true positives).

Recall is a measure how well out model finds the correct sign on an image. As an example: Out of all stop signs in the data set, how likely is it that our model finds them?

If we want to increase precision, our model must be very careful when picking images as every wrongly predicted class decreases precision. On the other hand, if it's too careful, it may miss some images with the right signs - which decreases recall. In Custom Vision there is a "Probability Threshold" slider we can adjust to change, how strict the model should be in classifying images: The lower the threshold, the less strict is the model (which means low precision but high recall). Here is an example table of different thresholds and the respective precision and recall values:

| Probability Threshold  | Precision | Recall |
| ------------- | ------------- | ------- |
| 10%  | 85.2% | 100% |
| 20% | 88.5% | 100% |
| 30% | 91.7% | 95.7% |
| 40% | 91.7% | 95.7% |
| 50% | 91.3% | 91.3% |
| 60% | 95.5% | 91.3% |
| 70% | 95.5% | 91.3% |
| 80% | 100% | 82.6% |
| 90% | 100% | 82.6% |
| 100% | 100% | 21.7% |

We can plot this table as a curve:

<img src="../images/precision_recall.png" width="400" />

A perfect (but in practice not achievable) model would have 100% precision and 100% recall at all different threshold values which would bend the curve to the right corner. 

Now suppose we have a second model with a different curve - how do we compare these curves and get a metric which curve is better? We could measure the **area under the curve** by calculating the integral between the lowest and highest precision-recall pair. In practice, we use an approximation by summing up the rectangles defined by precision multiplied with recall at a certain threshold. And this is called "Average Precision" (AP).

## Download and convert the model

1) Click on the "Export" button in the header. If this button is grey, you didn't select a "compact" model when creating the project. In the modal window, you'll see several model formats you can choose to download:

<img src="../images/customvision_8.png" width="400" />

2) Download the "TensorFlow" model (don't pick the "ONNX Windows ML" option). Unzip the package you get, you should see two files, `model.pb` and `labels.txt`. 

3) We have to convert this model into a format our car can use. We'll use the [tensorflow-onnx model converter](https://github.com/onnx/tensorflow-onnx) to do this. Run the following script on your PC (assuming TensorFlow is already installed)

```
pip install -U tf2onnx
```

Now run the following command. Make sure you're in the same directory as `model.pb`

```
python -m tf2onnx.convert --input model.pb --output model.onnx --inputs "Placeholder:0" --outputs "loss:0"
```

You should get a new file called `model.onnx`. This is the model we'll use for our car.

## Write a custom model preprocessor

Before an input image can be handled by the model, it has to be preprocessed. When you use the `AutTrainer` module and the `train()` method, this preprocessing is done for you automatically. When you use an external model, we have to bring the image into the right format so the model can use it. 

Let's take a closer look at the `model.onnx` file we just created. With a tool called [Netron](https://github.com/lutzroeder/netron) we can inspect how the model looks like:

<img src="../images/customvision_9.png" width="400" />

You see at "Placeholder:0" that our model.onnx expectes our images to be in the format [224x224x3] - this is not what we get from our car camera. To resize the image, create a new file called `trafficsign_sample.py` and add the following code:

```python
from autcar import Camera, Model
import numpy as np
import cv2

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
```

Here we create a sub class `OwnModel` which inherits from the `Model` base class. We overwrite the `preprocess` method which is transforming the image before feeding it into the model. Step by step, here is what happens:

- We get an RGB image, so the red channel is the first array. Our Custom Vision model expects images in the BGR format. Therefore, we split the channels and glue them together again in the expected order.
- Next we get the height (h) and width (w) of our image and get the smaller edge with `min()`. The next three lines crop the image along the larger edge so that we get a square image which is then resized to the size 224x224
- Finally, we return the resized BGR image

Now let's execute this model. We make use of our `Driver` class here even though for testing we don't really want the car to drive but just take a look at the real-time model predictions. Add the following code after the `OwnModel` class definition:

```python
model_trafficsigns = OwnModel("model.onnx", execution_interval=2, name="traffic_model")

def execute(model_predictions: dict, car: Car, variables: dict):
    print(model_predictions["traffic_model"][0])

driver = Driver(model_trafficsigns, None, cam, execution_function=execute, execution_interval=3)
driver.start()
```

First, we create a model object with our `model.onnx` file. We also define that this model should be executed every two seconds (note that there is a different execution interval for the model and the execution function). We also assign a name to this model.

Next we define a function `execute` which will be handed over to the `Driver` class. This standard function will get three arguments by `Driver`, a dictionary of predictions, the car object and a variables dictionary. `model_predictions` contains all predictions of all models we execute, in this case we just have one. In `execute` we print the results of the model predictions. The result is a list of the last 5 predictions the model made, index 0 holds the most recent predictions as an integer value. If we want to map back this integer value to a label, take a look into the `labels.txt` file you downloaded earlier:

<img src="../images/customvision_10.png" width="300" />

So if our model predicts "0", it means no traffic sign was detected.

Finally, run this script:

```
python trafficsign_sample.py
```

Our car is now capturing data from the camera and prints the predictions on the console (0 = No sign detected, 1 = major road sign detected, 2 = stop sign detected). Place the car in front of a sign to see how the predictions change.
