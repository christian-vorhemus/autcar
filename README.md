<img src="images/autcar_logo.png" width="400" />

This is the source code for the AutCar project - Build your own autonomous driving toy car.

<img src="images/autcar_modelone.png" width="400" />

[![Total alerts](https://img.shields.io/lgtm/alerts/g/christian-vorhemus/autcar.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/christian-vorhemus/autcar/alerts/)
[![Language grade: JavaScript](https://img.shields.io/lgtm/grade/javascript/g/christian-vorhemus/autcar.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/christian-vorhemus/autcar/context:javascript)


**Important**: This code is heavily under development right now, you'll certainly find bugs and shortcomings. Please create issues or contribute to the code.


## Getting started

### 1) Hardware Prerequisites

Make sure you have the following components ready:
1) A Raspberry Pi (any version is fine)
2) An expansion board for Raspberry Pi with two motors (e.g. [this](https://www.conrad.at/de/raspberry-pi-erweiterungs-platine-rb-moto2-raspberry-pi-raspberry-pi-a-b-b-1274197.html?ef_id=CjwKCAiA7vTiBRAqEiwA4NTO691Q8BTeqVSsY1307ua5BOyQi7aVhCghTbuAEv_ywCLANwHPqgztlBoC1lQQAvD_BwE:G:s&gclid=CjwKCAiA7vTiBRAqEiwA4NTO691Q8BTeqVSsY1307ua5BOyQi7aVhCghTbuAEv_ywCLANwHPqgztlBoC1lQQAvD_BwE) one)
3) A camera (either an USB webcam or a Raspberry Pi cam)
4) A battery to run your Raspberry
5) A chassis to place all components (you can make your own by 3D printing it, make one out of wood or cardbox)

### 2) Software setup

1) Prepare a SD card with [Raspian](https://www.raspberrypi.org/downloads/) for your Raspberry Pi
2) Python is already pre-installed on Raspian, additionally install OpenCV by typing
```
sudo apt-get install python-opencv
```
3) Make sure you can communicate with your Raspberry and your main PC, for example by [enabling the SSH server](http://raspberrypiguide.de/howtos/ssh-zugriff-unter-raspbian-einrichten/) on your Raspberry Pi.
4) Clone this repository on your Raspberry Pi **and** your main PC
```
git clone https://github.com/christian-vorhemus/autcar.git
```
5) CD to the autcar folder and install the requirements on your Raspberry Pi and main PC by typing
```
pip install -r requirements.txt
```

### 3) Run sample client on your Raspberry Pi
1) Run the following script (see /src folder) in your Raspberry Pi
```
python main.py
```
Now your Raspberry is listening on Port 8089 and 8090 for incoming connections from another PC.

### 4) Run server on your main PC
1) Switch to the /src/autcar/web directory. Them, run the command
```
python server.py
```
2) Open a browser and go to http://localhost:8080
3) Click on the red "Not connected" button to connect to your car
4) Press the green "Forward" button in the "Car control" window
5) The motors should now start moving

## Collect training data for the machine learning model
- Coming soon -

## Train a machine learning model with CNTK
- Coming soon -
