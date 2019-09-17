try:
    import RPi.GPIO as GPIO
except Exception as e:
    print("Warning: RPi.GPIO could not be loaded")
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

    def __init__(self, model='one', switch_left_right=False):
        """
        Use this object to control the motor of the car

        @param model: Selects which model type should be used.
        @param switch_left_right: Should be set to True when you notice that the car confuses left and right commands
        """
        self.__tright = None
        self.__moving_right = False
        self.__tleft = None
        self.__moving_left = False
        self._model = model
        GPIO.setmode(GPIO.BOARD)
        self.__reset_pins()
        self.__switch_left_right = switch_left_right
        self.current_command = {'type': 'started'}

    def __reset_pins(self):
        time.sleep(0.1)
        for pin in self.__control_pins_left:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, 0)

        for pin in self.__control_pins_right:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, 0)

    def __right_motor(self, direction, delay):
        while True:
            if(self.__moving_right == False):
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
            if(self.__moving_left == False):
                break
            for halfstep in range(len(self.__sequence)):
                for pin in range(4):
                    if(direction == 1):
                        GPIO.output(self.__control_pins_left[pin], self.__sequence[halfstep][pin])
                    else:
                        GPIO.output(self.__control_pins_left[pin], self.__sequence[len(self.__sequence)-halfstep-1][pin])
                time.sleep(delay)

    def __reset_movement(self):
        try:
            self.__moving_right = False
            self.__moving_left = False
        except Exception as e:
            pass
        self.__reset_pins()

    def move(self, direction = "forward", speed = "medium"):
        """
        Move the car straight either forward or backwards in different speed

        @param direction: One of the following strings: "forward" or "backwards"
        @param speed: One of the following strings "slow", "medium" or "fast"
        """

        self.__reset_movement()
        time.sleep(0.1)

        if(direction == "forward"):
            di = 0
        else:
            di = 1

        if(speed == "fast"):
            motor_delay = 0.0007
        elif(speed == "medium"):
            motor_delay = 0.001
        elif(speed == "slow"):
            motor_delay = 0.005
        else:
            motor_delay = 0.001

        self.__tright = Thread(target = self.__right_motor, args = (di,motor_delay,))
        self.__tleft = Thread(target = self.__left_motor, args = (di,motor_delay,))
        self.__moving_left = True
        self.__moving_right = True
        self.__tright.start()
        self.__tleft.start()
        self.current_command = {'type' : 'move', 'direction' : direction, 'speed': speed}

    def stop(self):
        """
        Stops the motors of the car
        """
        try:
            self.__moving_right = False
            self.__moving_left = False
        except Exception as e:
            pass
        self.__reset_pins()
        self.current_command = {'type' : 'stop'}

    def right(self, style = "medium", direction = "forward"):
        """
        Move the car to the right.

        @param style: Defines how fast the car moves to the right. One of the following strings: "light", "medium", "harsh"
        @param speed: Defines the direction. One of the following strings: "forward" or "backwards"
        """

        self.__reset_movement()
        time.sleep(0.1)

        if(direction == "backwards"):
            di = 1
        else:
            di = 0

        __speed_right = 0.001
        __speed_left = 0.001

        if(style == "medium"):
            __speed_right = 0.005
            __speed_left = 0.001
        elif(style == "harsh"):
            __speed_right = 0.005
            __speed_left = 0.0006
        elif(style == "light"):
            __speed_right = 0.002
            __speed_left = 0.001

        time.sleep(0.1)

        if(self.__switch_left_right):
            temp = __speed_right
            __speed_right = __speed_left
            __speed_left = temp

        self.__tright = Thread(target = self.__right_motor, args = (di,__speed_right,))
        self.__tleft = Thread(target = self.__left_motor, args = (di,__speed_left,))

        self.__moving_left = True
        self.__moving_right = True
        self.__tright.start()
        self.__tleft.start()
        self.current_command = {'type' : 'right', 'direction' : direction, 'style': style}

    def left(self, style = "medium", direction = "forward"):
        """
        Move the car to the left

        @param style: Defines how fast the car moves to the left. One of the following strings: "light", "medium", "harsh"
        @param speed: Defines the direction. One of the following strings: "forward" or "backwards"
        """

        self.__reset_movement()
        time.sleep(0.1)

        if(direction == "backwards"):
            di = 1
        else:
            di = 0

        __speed_right = 0.001
        __speed_left = 0.001

        if(style == "medium"):
            __speed_right = 0.001
            __speed_left = 0.005
        elif(style == "harsh"):
            __speed_right = 0.0006
            __speed_left = 0.005
        elif(style == "light"):
            __speed_right = 0.001
            __speed_left = 0.002

        time.sleep(0.1)

        if(self.__switch_left_right):
            temp = __speed_right
            __speed_right = __speed_left
            __speed_left = temp

        self.__tright = Thread(target = self.__right_motor, args = (di,__speed_right,))
        self.__tleft = Thread(target = self.__left_motor, args = (di,__speed_left,))

        self.__moving_left = True
        self.__moving_right = True
        self.__tright.start()
        self.__tleft.start()
        self.current_command = {'type' : 'left', 'direction' : direction, 'style': style}