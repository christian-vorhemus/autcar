# Train a machine learning model locally

In this tutorial we are going to take a closer look at that happens when we train a machine learning model. As a prerequisite make sure you have some training data collected as described [here](4_AutCar_General.md#create-training-data) or [here](3_Autonomous_Driving.md#2-capture-training-data).

## The AutTrainer module

Suppose have training data, now let's see how we can use it. AutCar offers a module called _AutTrainer_ which provides several methods to help you with training a model. The first method we'll take a look at is `create_balanced_dataset()`. This method does two things: First, it balances the dataset. It's unlikely that we use all commands (move, left, right...) the same number of times. But to create an unbiased model, all classes should appear roughly uniformly. `create_balanced_dataset()` is **upsampling** the data by simply copying underrepresented classes. Second, the method also splits our data into a **training** and a **test** set. While data in the training set is used to train the model, images in the test set are only used to evaluate model performance. And for a meaningful evaluation we have to use data, the model has not seen before durng training.

To create a balanced dataset, use the following code:

   ```python
  from autcar import Trainer

  trainer = Trainer()
  trainer.create_balanced_dataset(input_folder_path = "path/to/trainingdata", output_folder_path = "path/to/trainingdata_balanced", train_test_split = 0.7)

  ```

The argument `input_folder_path` tells the trainer where our folder created by _AutCapture_ is located. `output_folder_path` can be an arbitrary path to location where the balanced dataset should be created. `train_test_split` tells our method how much training images should go into the training set (in this case 70%). 

Let's use another method to see which and how many labels we have in our dataset. Add the following code:

   ```python
  labels = trainer.get_classes("path/to/trainingdata_balanced")
  print(labels)

  ```
  
This should print the labels we have in the dataset and the corresponding amount on the console, for example:

  ```
  {'move_medium_forward': 195, 'right_medium_forward': 198, 'left_medium_forward': 186}
  ```
  
In this example we just used three commands to drive our car: "move_medium_forward", "right_medium_forward" and "left_medium_forward". In total, there are 12 possible commands to control the car. Each of these commands has a number assigned to it as shown in the image below:

<img src="../images/controls.png" width="800">

## Convolutional Neural Network basics

When a machine learning model makes predictions, it doesn't output text. It always outputs number. We have to map these numbers back to useful labels which means, if the model outputs for example "6" the corresponding command is "move the car forward with medium speed".

Before we get predictions, we have to define our model and we'll use a Convolutional Neural Network (CNN) to do this job. Look at the images below: These are typical examples of what the car sees and the text below tells us what command we would expect to be executed when an image like this appears infront of the car:

  <img src="../images/movements.png" width="500">

  CNNs are sequences of operations stacked on each other. We start with an input layer that represents our image. Next, the image goes through an "Convolution layer" (in Keras the corresponding method is called "Conv2D"). Then the images is shrinked in a layer called "MaxPool2D". This happens a few times until the 2D images is converted to a 1D vector through "Flatten". In the end we have a fully-connected 1D "Dense" layer consisting of 12 neurons representing the 12 possible movements the car can make. This vector holds probabilities, so for example if the output vector for one image looks like this `[0.05,0.05,0.8,0.1,0,0,0,0,0,0,0,0]` the highest probability is at index 2 which is equivalent to the command "left medium backwards".
  
  In Keras, the CNN as described above can be implemented as follow:

  ```python
  from keras.models import Sequential
  from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, InputLayer
  
  model = Sequential([
    InputLayer(input_shape=[3,168,224]),
    Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu'),
    MaxPool2D(pool_size=8, padding='same'),
    Conv2D(filters=48, kernel_size=3, strides=1, padding='same', activation='relu'),
    MaxPool2D(pool_size=5, padding='same'),
    Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
    MaxPool2D(pool_size=3, padding='same'),
    Conv2D(filters=32, kernel_size=5, strides=1, padding='same'),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(12, activation='softmax')
  ])
  ```
  
  Let's take a closer look at the single steps here: "InputLayer" needs the shape of out image, the standard size we use is 3 channels (Red, Green, Blue = RGB), 168 pixel height and 224 pixel width. Next, the convolutional operation takes place. Here, we define a set of filters which are sliding over the image whereby the pixel values of the filter and the underlying part of the image are multiplied. The filter generates a large "pulse" when it moves over regions that are similar to the filter pattern (note: In the image below these results are represented as grey boxes in the filter map, in a real CNN, these values would be numbers)
  
  <img src="../images/kernels.gif" width="500">
  
  In the "MaxPool2D" layer, the created feature map is resized to a smaller matrix:
  
  <img src="../images/pooling.png" width="500">

Finally, 
