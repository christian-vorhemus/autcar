# Hardware Assembly

In this chapter you'll see how to assembly all the hardware parts to construct your car!

### 1) Prepare all parts
Make sure you have all parts next to you before you start
- The chassis (underbody)
- 2x motors
- 2x motor-wheel connectors
- Battery pack
- Expansion board to connect motors and Raspberry Pi
- 2x gummed back wheels
- 2x front wheels (including axle)
- Raspberry Pi 3
- Raspberry Pi Camera
- 2x screws

Additionally, you will need the following:
- Crosstip screwdriver
- Scissors or a stanley knife
- File (optional)

First, remove the brim from the motor-wheel connectors. You can do this either by hand or with the help of a scissor

<img src="../images/A1.jpg" width="400">

Do the same with the chassis:

<img src="../images/A2.jpg" width="400">

Check if you can put one of the motors into the motor bracket. You should feel a certain resistance but the brackets shouldn't bend too much. You may want to use a file and carefully abrade the little bulge in the motor bracket:

<img src="../images/A3.jpg" width="400">

### 2) Front wheels
Take the wheel with axle and put it through the front part of the chassis. Then, with some pressure put the other wheel on it. Again, you may want to **use a file** to carefully abrade the axle which makes it easier to assemble the front wheel part.

<img src="../images/A4.jpg" width="400">

### 3) Back wheels
Take the motor-wheel connector and place it in the wheel so the rectangular part is visible from the back side:

<img src="../images/A5.jpg" width="400">

Next, screw the motor-wheel connector to the wheel.

<img src="../images/A6.jpg" width="400">

Repeat with the second back wheel and plug the motor-wheel connector on the motors.

<img src="../images/A7.jpg" width="400">

Afterwards, plug both motors into the motor bracket.

<img src="../images/A8.jpg" width="400">

### 4) Connect all parts
Gently place the battery pack in the chassis and put the Raspberry Pi on top of it. There is no need to fasten it. Now, plug the motors into the expansion board and put the expansion board on the GPIO pins as shown in the image below:

<img src="../images/A9.jpg" width="400">

Last, plug the camera into the Raspberry Pi Camera CSI interface and fix it in the front mounting. Take the built-in USB cable of the battery pack, plug it into the Raspberry Pi power plug and finally bend all cords together.

<img src="../images/A10.jpg" width="400">

Your car is ready, perfect! Next, let's install the software, click [here](https://github.com/christian-vorhemus/autcar/blob/master/docs/Software_Setup.md)
