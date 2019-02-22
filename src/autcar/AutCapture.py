from threading import Thread
import time
import os

class Capture:

    def __init__(self, car, cam, folder_name = "autcar_training", capture_interval = 2):
        """
        A capture object can be used to record training data while the car is driving

        @param car: The car object which is used to retriece the current commands
        @param cam: The cam object which is used to take snapshots of the track
        @param folder_name: The folder where training images and commands are stored
        @param capture_interval: Defines how often pictures are taken while recording. Default is 2 seconds
        """
        self.__folder_name = folder_name
        self.__car = car
        self.__cam = cam
        self.__proc = Thread(target=self.__record_data)
        self.__stop_recording = False
        self.__capture_interval = capture_interval

    def __record_data(self):

        if not os.path.exists(self.__folder_name):
            os.mkdir(self.__folder_name)

        while True:
            if(self.__stop_recording):
                break
            
            self.__cam.take_snapshot(self.__folder_name, self.__car)
            time.sleep(self.__capture_interval)    

    def start(self):
        self.__proc.start()

    def stop(self):
        self.__stop_recording = True