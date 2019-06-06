import cv2
import pickle
import numpy as np
import struct
import base64
import socket
import threading
import os
import time
import subprocess

class Camera:

    def __init__(self, connect_camera = False, host = '', port = 8089, rotation = None, capture_interval = 1):
        """
        A camera object which is used to capture single images from a camera or start a live stream. It uses OpenCV under the hood.

        @param connect_camera: If True, a socket connection is opened to the address and sport specified in the constructor
        @param host: Defines the name of the host for camera strean. Default is localhost
        @param port: Defines the port the camera is using for sending live images. Default to 8089
        @param rotation: Defines if camera images should be rotated. Default is none, use -1 for 180 degree rotation
        """
        self.__frame = None
        self.host = host
        self.port = port
        self.__rotation = rotation
        self.__nosignal = True
        self.__proc = threading.Thread(target=self.__listen_socket)
        self.__stop_sending = False
        self.__capture_interval = capture_interval
        # Load Rasperry Pi Cam kernel module bcm2835-v4l2
        try:
            subprocess.check_call("sudo modprobe bcm2835-v4l2", shell=True)
        except:
            print("Warning: Couldn't load bcm2835-v4l2 kernel module")
        self.__cam = cv2.VideoCapture(0)
        if(connect_camera):
            threading.Thread(target=self.__frame_updater).start()

    def get_frame(self):
        """
        Returns the current camera frame as byte object
        """
        return self.__frame


    def read(self):
        """
        Returns the frame of the OpenCV VideoCapture buffer
        """
        ret, frame = self.__cam.read()
        if(self.__rotation != None):
            frame = cv2.flip(frame, self.__rotation)
        return ret, frame


    def __frame_updater(self):
        clientsocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        try:
            clientsocket.connect((self.host, self.port))
        except Exception as e:
            print("Could not connect to remote machine: "+str(e))
            self.__nosignal = True
            return

        data = b""
        self.__nosignal = False
        payload_size = struct.calcsize("<L")
        while True:
            try:
                while len(data) < payload_size:
                    data += clientsocket.recv(4096)

                frame_size = struct.unpack("<L", data[:payload_size])[0]
                data = data[payload_size:]

                while len(data) < frame_size:
                    data += clientsocket.recv(16384)

                frame_data = data[:frame_size]
                data = data[frame_size:]

                img = base64.b64decode(frame_data)
                npimg = np.fromstring(img, dtype=np.uint8)
                frame = cv2.imdecode(npimg, 1)

                #cv2.imwrite("img.jpg", frame)

                ret, jpeg = cv2.imencode('.jpg', frame)
                self.__frame = jpeg.tobytes()


            except (socket.error,socket.timeout) as e:
                # The timeout got reached or the client disconnected. Clean up the mess.
                print("Cleaning up: ",e)
                try:
                    clientsocket.close()
                except socket.error:
                    pass
                nosignal = True
                break


    def start(self):
        """
        Starts a live streaming camera session. Should be called on the device which wants to broadcast
        """
        self.__proc.start()

    def stop(self):
        """
        Stops the camera live stream, closes the socket
        """
        self.__stop_sending = True

    def __listen_socket(self):
        serversocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        serversocket.bind((self.host, self.port))
        serversocket.listen(10)
        print('Camera socket now listening on ' + self.host + ":" + str(self.port))

        conn, addr = serversocket.accept()
        print("New client connection")
        last_timestamp = time.time()

        while not self.__stop_sending:

            ret, frame = self.read()
            current_time = time.time()

            if(current_time - last_timestamp > self.__capture_interval):
                last_timestamp = current_time
                frame = cv2.resize(frame,(200, 150))
                encoded, buffer = cv2.imencode('.jpg', frame)
                b_frame = base64.b64encode(buffer)
                b_size = len(b_frame)
                print("Frame size = ", b_size)
                try:
                    conn.sendall(struct.pack("<L", b_size) + b_frame)
                except socket.error as e:
                    print("Socket Error: "+str(e))
                    self.__stop_sending = True
                    conn.close()
                    serversocket.close()

            