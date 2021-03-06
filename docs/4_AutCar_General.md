# Overview of AutCar

This tutorial gives some hands-on experience with the AutCar library more information about the training process and useful methods. As a prerequisite, make sure that your car is [assembled](https://github.com/christian-vorhemus/autcar/blob/master/docs/1_Hardware_Assembly.md), the [software is installed](https://github.com/christian-vorhemus/autcar/blob/master/docs/2_Software_Setup.md) and you have an open SSH connection to your car.

## The library

The AutCar library consists of independent modules depicted in the image below:

<img src="../images/autcar_stack.jpg" width="700">

- **AutCamera** offers an abstraction layer to access the Raspberry Pi camera. Under the hood, OpenCV `cv2.VideoCapture` is used to capture frames. AutCamera uses the first video capturing device it finds (index 0). If you have multiple cameras attached, you may change the device manually [here](https://github.com/christian-vorhemus/autcar/blob/master/src/autcar/AutCamera.py#L34).
- **AutCapture** is used to create training data. It needs a camera and a car object to work as it takes the current camera frame and the current car commands for data recording.
- **AutCar** is used to control the motors. For Model One, two standard stepper motor are used controlled by the GPIO Pins `[11,12,13,15]` for the left motor and `[16,18,22,7]` for the right motor.
- **AutDriver** takes one or several machine learning model files in ONNX format and an execution function and constantly performs the actions defined in the execution function based on the machine learning model predictions.
- **AutRemoteController** offers functions to open up a connection between your PC and the car over sockets.
- **AutTrainer** provided some helper functions to create a balanced dataset or to train a model.

## Start the engines

Let's write a simple script that controls the motors. On your car, create a python script called `motor_test.py` in the same folder where your main `autcar` directory is located. 

  ```
  nano motor_test.py
  ```

Copy the following code into the file:

  ```python
  from autcar import Car
  import time
  
  car = Car()
  
  car.move()
  time.sleep(3)
  car.stop()
  ```
  
  Then execute the code with
  
   ```
   python3 motor_test.py
  ```
  
  This script creates a car object and calls the `move()` method which tells the car to drive forward. Then, we wait for three seconds until we stop the car.
 
 Let's play around with this methods even more: The following code generates a random integer between 0 and 3 and controls the car based on this number. This happens 5 times until the car is stopped:
  ```python
  from autcar import Car
  import random
  import time
  
  car = Car()
  
  for x in range(5):
    cmd = random.randint(0,4)
    print(cmd)
    if(cmd == 0):
      car.move("forward")
    elif(cmd == 1):
      car.move("backwards")
    elif(cmd == 2):
      car.left("light")
    else:
      car.right("light")
    time.sleep(3)
    
  car.stop()
  ```
  You can pass arguments to the methods to specify in more detail what to do. For example, the first argument in `move()` method tells the car in which direction to move (supported are "forward" and "backwards"). The first argument in `left()` or `right()` tells the car how strong it should change the direction (supported are "light", "medium" and "harsh").

## Listen for commands and send commands

It's nice to control the car directly on the car - but sometimes we want to send data from an external computer to the vehicle - that's what the _AutRemoteController_ module is for. Let's write two things now: A script that runs on your car listening for external commands and a script for your PC sending commands to the car.

Create a new file `rc_test.py` on your car. Copy the following code into it:

  ```python
  from autcar import RemoteController, Car
  import time
  
  car = Car()
  rc = RemoteController()
  rc.listen()

  try:
    while True:
      cmd = rc.get_cmds()
      if(cmd == "forward"):
        car.move()
      elif(cmd == "stop"):
        car.stop()
  except KeyboardInterrupt:
    rc.close()
  ```

With `rc = RemoteController()` we create a new RemoteController object. This object holds all the methods and properties we need to establish a connection between the car and the PC. With `rc.listen()` we tell the car to listen for incoming commands. Since we don't want to just process one command, we wrap the code into a while loop to continuously fetch new commands from the socket. In this example we just allow two commands: Move the car forward or stop the car.

Next we're going to write a simple script that allows users to enter commands on the command line which are then sent to the car. Create a new file on your PC and copy the following code into it:

  ```python
  from autcar import RemoteController
  
  rc = RemoteController(host = "192.168.1.1")
  rc.connect()
  
  print("Enter 'f' to drive forward or 's' to stop the car:")
  
  try:
    while True:
      cmd = input("> ")
      if(cmd == "f"):
        rc.send_cmd("forward")
      elif(cmd == "s"):
        rc.send_cmd("stop")
      else:
        print("Unknown command, enter 'f' or 's'")
  except KeyboardInterrupt:
    rc.close()
  ```
  Change the IP address in line 3 to the IP address of your car and execute this script on your PC. You should now be able to enter either "f" or "s" to control the car.

## Create a live stream from your car

Let's take a look at some code how we can open a connection from our PC to the camera of our car. Create a file called `camera_test.py` on your car and copy and paste the following content:
  ```python
  from autcar import Camera

  cam = Camera(rotation=-1)
  cam.start()
  ```
  
  Execute the code with
   ```
   python3 camera_test.py
  ```
  
  Your car is now listening on port 8089 for interested live stream viewers. The `rotation=-1` argument flips the images by 180 degrees. This is necessary because the camera is mounted reversed in the AutCar development kit.
  
  On your PC, go to your autcar src folder and enter
   ```
   python autcar/web/server.py
  ```
  to start the AutCar Control Board server. Open a browser, type in http://localhost:8080 and enter the IP address of your car in the right upper corner. Click "Connect" and you should see the live stream in a second.

## Create training data

If our car should drive autonomously, we have to teach it how to drive. And we want the car to learn how to drive from camera images only. To do so, we have to manually drive the car and capture all the images the car sees including the commands we used to control the car. Luckily, the _AutCapture_ module does most of the job for us, but let's take a look at what happens here from scratch.

Create a new file called `capture_test.py` and add the following code

  ```python
  from autcar import Car, Camera, Capture
  import time

  car = Car()
  cam = Camera(rotation=-1)
  cap = Capture(car, cam, capture_interval=1, folder_name="trainingdata")
  
  cap.start()
  car.move()
  time.sleep(3)
  car.left()
  time.sleep(3)
  car.stop()
  time.sleep(2)
  cap.stop()
  ```

  AutCapture needs a camera object and a car object. It saves the pictures and commands in a newly created folder named "trainingdata" on your car. `capture_interval=1` tells our capture object to record data every second. 
  
  Execute the script with
  ```
  python3 capture_test.py
  ```
  
  Your car should move forward, to the left and then stop. Now check if a newly created folder called "trainingdata" is available.
  
  ```
  ls
  ```
  
  Copy this folder to your PC by entering the following command on your PC console:
  
  ```
  scp -r pi@192.168.1.1:/home/pi/autcar/trainingdata .
  ```
  
Make sure to add the right path for your "trainingdata" folder and change the IP address accordingly.

Now let's take a closer look what's in the folder. You should see several images starting with "0_car_snapshot.png". Additionally, a CSV file named "training.csv" is created. Open the file. There are two columns available: The first column contains the name of the image and the second column the corresponding command that was active when the picture was captured as a JSON string. As expected, we see the "move forward" command in the first three rows followed by the "left" command and "stop" afterwards. 

<img src="../images/autcar_commands.png" width="400">

To get real training data, create a track  as described [here](3_Autonomous_Driving.md#1-create-track), place your car on the track and control the car manually through the circuit while capturing images. Do this for several rounds (we recommend at least 10) to get representing training data.
