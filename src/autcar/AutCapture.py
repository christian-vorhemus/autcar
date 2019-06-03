from threading import Thread
import subprocess
import time
from autcar import Camera
import cv2
import os

class Capture:

    def __init__(self, car, camera, capture_interval = 2, folder_name = "autcar_training"):
        """
        A capture object can be used to record training data while the car is driving

        @param car: The car object which is used to retrieve the current commands
        @param folder_name: The folder where training images and commands are stored
        @param capture_interval: Defines how often pictures are taken while recording. Default is 2 seconds
        @param camera: A camera object used to take training pictures
        """
        self.__folder_name = folder_name
        self.__car = car
        self.__cam = camera
        self.__frame = None
        self.__proc = Thread(target=self.__record_data)
        self.__stop_recording = False
        self.__capture_interval = capture_interval
        self.__counter = 0
        self.__last_timestamp = 0

    def __save_frame(self, frame, folder_name, description):
        img_name = folder_name + "/" + str(self.__counter) + "_car_snapshot.png"
        cv2.imwrite(img_name, frame)
        with open(folder_name+'/training.csv', 'a') as f:
            f.write(str(self.__counter) + "_car_snapshot.png;" + str(description) + "\r\n")
            self.__counter = self.__counter + 1

    def __record_data(self):

        if not os.path.exists(self.__folder_name):
            os.mkdir(self.__folder_name)

        self.__last_timestamp = time.time()

        while True:
            if(self.__stop_recording):
                break

            # We constantly read new images from the cam to empty the VideoCapture buffer
            ret, frame = self.__cam.read()
            self.__frame = frame
            current_time = time.time()

            if(current_time - self.__last_timestamp > self.__capture_interval):
                self.__save_frame(self.__frame, self.__folder_name, self.__car.current_command)
                self.__last_timestamp = current_time

    def start(self):
        """
        Start the recording of camera images and car movements. Results are saved locally
        """
        self.__proc.start()

    def stop(self):
        """
        Stops the recoding of images and car movements
        """
        self.__stop_recording = True