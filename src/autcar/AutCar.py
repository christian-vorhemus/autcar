import RPi.GPIO as GPIO
import time
from threading import Thread

class Car:

    __control_pins_left = [11,12,13,15]
    __control_pins_right = [16,18,22,7]

    __sequence = [
        [1,0,0,0],
        [1,1,0,0],
        [0,1,0,0],
        [0,1,1,0],
        [0,0,1,0],
        [0,0,1,1],
        [0,0,0,1],
        [1,0,0,1]
    ]

    def __init__(self, model=1):
        self.__stop_right = False
        self.__stop_left = False
        self._model = model
        GPIO.setmode(GPIO.BOARD)
        self.__reset_pins()
        self.current_command = {'type': 'started'}

    def __reset_pins(self):
        time.sleep(0.1)
        for pin in self.__control_pins_left:
            a = 1
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, 0)

        for pin in self.__control_pins_right:
            a = 1
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, 0)

    def __right_motor(self, direction, delay):
        while True:
            if(self.__stop_right):
                break
            for halfstep in range(len(self.__sequence)):
                for pin in range(4):
                    if(direction == 1):
                        GPIO.output(self.__control_pins_right[pin], self.__sequence[len(self.__sequence)-halfstep-1][pin])
                    else:
                        GPIO.output(self.__control_pins_right[pin], self.__sequence[halfstep][pin])
                time.sleep(delay)


    def __left_motor(self, direction, delay):
        while True:
            if(self.__stop_left):
                break
            for halfstep in range(len(self.__sequence)):
                for pin in range(4):
                    if(direction == 1):
                        GPIO.output(self.__control_pins_left[pin], self.__sequence[halfstep][pin])
                    else:
                        GPIO.output(self.__control_pins_left[pin], self.__sequence[len(self.__sequence)-halfstep-1][pin])
                time.sleep(delay)


    def move(self, direction = "forward", speed = "medium"):
        """
        Move the car straight either forward or backwards in different speed

        @param direction: One of the following strings: "forward" or "backwards"
        @param speed: One of the following strings "slow", "medium" or "fast"
        """

        self.stop()
        time.sleep(0.1)
        self.__stop_right = False
        self.__stop_left = False

        if(direction == "forward"):
            di = 1
        else:
            di = 0

        if(speed == "fast"):
            motor_delay = 0.0006
        elif(speed == "medium"):
            motor_delay = 0.001
        elif(speed == "slow"):
            motor_delay = 0.005
        else:
            motor_delay = 0.001

        tright = Thread(target = self.__right_motor, args = (di,motor_delay,))
        tleft = Thread(target = self.__left_motor, args = (di,motor_delay,))
        tright.start()
        tleft.start()
        self.current_command = {'type' : 'move', 'direction' : direction, 'speed': speed}

    def stop(self):
        """
        Stops the car
        """
        self.__stop_left = True
        self.__stop_right = True
        self.__reset_pins()
        self.current_command = {'type' : 'stop'}


    def right(self, style = "medium", direction = "forward"):
        """
        Move the car to the right.

        @param style: Defines how fast the car moves to the right. One of the following strings: "light", "medium", "harsh"
        @param speed: Defines the direction. One of the following strings: "forward" or "backwards"
        """

        self.stop()
        time.sleep(0.1)
        self.__stop_left = True
        self.__stop_right = True

        if(direction == "backwards"):
            di = 0
        else:
            di = 1

        __speed_right = 0.001
        __speed_left = 0.001

        if(style == "medium"):
            __speed_right = 0.005 
            __speed_left = 0.0006
        elif(style == "harsh"):
            __speed_right = 0.005
            __speed_left = 0.0006
        elif(style == "light"):
            __speed_right = 0.005
            __speed_left = 0.001

        time.sleep(0.1)

        __tright = Thread(target = self.__right_motor, args = (di,__speed_right,))
        __tleft = Thread(target = self.__left_motor, args = (di,__speed_left,))

        self.__stop_left = False
        self.__stop_right = False

        __tright.start()
        __tleft.start()
        self.current_command = {'type' : 'right', 'direction' : direction, 'style': style}


    def left(self, style = "medium", direction = "forward"):
        """
        Move the car to the left

        @param style: Defines how fast the car moves to the left. One of the following strings: "light", "medium", "harsh"
        @param speed: Defines the direction. One of the following strings: "forward" or "backwards"
        """

        self.stop()
        time.sleep(0.1)
        self.__stop_left = True
        self.__stop_right = True

        if(direction == "backwards"):
            di = 0
        else:
            di = 1

        __speed_right = 0.001
        __speed_left = 0.001

        if(style == "medium"):
            __speed_right = 0.0006 
            __speed_left = 0.005
        elif(style == "harsh"):
            __speed_right = 0.0006
            __speed_left = 0.005
        elif(style == "light"):
            __speed_right = 0.001
            __speed_left = 0.005

        time.sleep(0.1)

        __tright = Thread(target = self.__right_motor, args = (di,__speed_right,))
        __tleft = Thread(target = self.__left_motor, args = (di,__speed_left,))

        self.__stop_left = False
        self.__stop_right = False

        __tright.start()
        __tleft.start()
        self.current_command = {'type' : 'left', 'direction' : direction, 'style': style}