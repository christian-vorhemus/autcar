import cv2
import pickle
import numpy as np
import struct
import socket
import threading
import os
import time
import subprocess

class Camera:

    def __init__(self, capture = False, host = 'localhost', port = 8089, rotation = None):
        """
        A camera object which is used to capture single images from a camera or start a live stream. It uses OpenCV under the hood.

        @param capture: If True, a camera socket is opened, default is false
        @param host: Defines the name of the host for camera strean. Default is localhost
        @param port: Defines the port the camera is using for sending live images. Default to 8089
        @param rotation: Defines if camera images should be rotated. Default is none, use -1 for 180 degree rotation
        """
        self.__frame = None
        self.host = host
        self.port = port
        self.__rotation = rotation
        self.__nosignal = True
        # Load Rasperry Pi Cam kernel module bcm2835-v4l2
        try:
            subprocess.check_call("sudo modprobe bcm2835-v4l2", shell=True)
        except:
            print("Warning: Couldn't load bcm2835-v4l2 kernel module")
        self.__cam = cv2.VideoCapture(0)
        if(capture):
            threading.Thread(target=self.frame_updater).start()

    def get_frame(self):
        if(self.__nosignal == False):
            return self.__frame
        else:
            fr = None
            return fr

    def read(self):
        ret, frame = self.__cam.read()
        if(self.__rotation != None):
            frame = cv2.flip(frame, self.__rotation)
        return ret, frame

    def frame_updater(self):
        clientsocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        try:
            clientsocket.connect((self.host, self.port))
        except:
            self.__nosignal = True
            return

        data = b""
        payload_size = struct.calcsize("L") 
        while True:
            while len(data) < payload_size:
                data += clientsocket.recv(100000)

            self.__nosignal = False
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("L", packed_msg_size)[0]

            while len(data) < msg_size:
                data += clientsocket.recv(100000)

            frame_data = data[:msg_size]
            data = data[msg_size:]

            # Could cause incompatibility between Python 2 and 3. remove "encoding" parameter potentially
            frame = pickle.loads(frame_data, encoding='latin1')
            ret, jpeg = cv2.imencode('.jpg', frame)
            self.__frame = jpeg.tobytes()

    def connect(self, host = 'localhost', port = 8089):
        clientsocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        clientsocket.connect((host, port))

        data = b""
        payload_size = struct.calcsize("L") 
        while True:
            while len(data) < payload_size:
                data += clientsocket.recv(4096)

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("L", packed_msg_size)[0]

            while len(data) < msg_size:
                data += clientsocket.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]

            # Could cause incompatibility between Python 2 and 3. remove "encoding" parameter potentially
            frame = pickle.loads(frame_data, encoding='latin1')
            # ret, jpeg = cv2.imencode('.jpg', frame)
            print("new image")
            return jpeg.tobytes()

    def listen(self, host = '', port = 8089):
        serversocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        serversocket.bind((host, port))
        serversocket.listen(10)
        print('Camera socket now listening on ' + host + ":" + str(port))

        while True:
            try:
                conn, addr = serversocket.accept()
                print("New client connection")
                while True:
                    ret, frame = self.__cam.read()
                    (w, h, c) = frame.shape
                    frame = cv2.resize(frame,(200, 150))
                    data = pickle.dumps(frame)
                    print("sending data...")
                    tosend = struct.pack("L", len(data))+data
                    conn.sendall(tosend)
                    time.sleep(3)
            except:
                continue
            