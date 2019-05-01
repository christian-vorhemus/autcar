# Autonomous Driving

This documents gives an overview how you can collect training data, train the car and start the autonomous driving mode.

### 1) Create track

Your car has to know when it is "on track" and when it leaves the track. Therefore, you need to prepare a circuit that has the following properties:
- The boundaries of the track should be clearly distinguishable from the ground.
- The right and left borders should have different colors.
Below, you see an example of a suitable track
<img src="../images/C1.jpg" width="400">

Also make sure that the track is wide enough for the car to be able to take bends well.

<img src="../images/C2.jpg" width="400">

Additionally, take the the following into account:
- Choose a room with uniform lighting. Backlighting can deteriorate the result.
- The total circuit size should be between 2x2 meters

### 2) Capture training data

1. On your Raspberry Pi, start the sample remote control script rc_sample.py
 ```
 python3 rc_sample.py
 ```

 Your car is now listening for commands.

2. On your PC, start the web client with
 ```
 python autcar/src/autcar/web/server.py
 ```

3. In the AutCar Control Board, connect to your car by entering the IP address and port in the upper right corner, click "Connect" and try if the connection works by clicking one of the control buttons.

4. Place the car on the circuit - we are ready to record training data now!

5. In the data recording section, click on "START". You should see a red dot flashing - this means we are recording data.

<img src="../images/D1.jpg" width="400">

6. Start driving your car manually by using the control buttons while the "REC" icon is flashing. Drive several rounds (we recommend to record at least 10 fully driven rounds) and when you're done, press the "STOP" button.

7. Stop the execution of the rc_sample.py script on your Raspberry by entering Ctrl+C and type
 ```
 ls
 ```
 You should see a new folder named "autcar_training". This folder contains images and a transcript of the commands you entered while driving the car.
 
 8. Copy the folder to your Desktop computer, we'll use the Secure Copy Protocol (SCP) for this. On your PC, open a command prompt and enter
  ```
 scp -r pi@192.168.1.101:/home/pi/autcar/src/autcar_training .
 ```
 Note: You **have to change the IP address** in the command above the the IP address of your car as well as the path to your autcar_training directory. The copying process may take a while
 
9. Look into the autcar_training folder, it should contain several images and a train.csv file.

<img src="../images/D1.jpg" width="400">

### 3) Train your model

- To do -

### 4) Test your model

- To do -

### 5) Run model on your AutCar

- To do -
