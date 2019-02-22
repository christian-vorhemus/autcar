import cv2
import pickle
import numpy as np
import struct
import socket
import threading
import os
import time

class Camera:

    def __init__(self, capture = False, host = 'localhost', port = 8089):
        self.__cam = cv2.VideoCapture(0)
        self.__counter = 0
        self.__frame = None
        self.host = host
        self.port = port
        self.__nosignal = True
        if(capture):
            #self.__nosignalframes = [open('/autcar/web/res/' + f + '.jpg', 'rb').read() for f in ['f1', 'f2', 'f3']]
            threading.Thread(target=self.frame_updater).start()

    def __saveFrame(self, frame, folder_name, description):
        img_name = folder_name + "/" + str(self.__counter) + "_car_snapshot.png"
        cv2.imwrite(img_name, frame)
        with open(folder_name+'/training.csv', 'a') as f:
            f.write(str(self.__counter) + "_car_snapshot.png;" + description + "\r\n")
            self.__counter = self.__counter + 1


    def take_snapshot(self, folder_name, car):
        ret, frame = self.__cam.read()
        carcmd = car.current_command
        print(carcmd)

        # Saving to file costs time, outsource it to a new thread
        threading.Thread(target=self.__saveFrame, args=(frame, folder_name, str(carcmd),)).start()


    def get_frame(self):
        if(self.__nosignal == False):
            return self.__frame
        else:
            fr = None
            return fr


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
                data += clientsocket.recv(4096)

            self.__nosignal = False
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("L", packed_msg_size)[0]

            while len(data) < msg_size:
                data += clientsocket.recv(4096)

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
            ret, jpeg = cv2.imencode('.jpg', frame)
            print("new image")
            return jpeg.tobytes()
            #cv2.imshow('frame', frame)
            #cv2.waitKey(1)


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
                    time.sleep(4)
                    ret, frame = self.__cam.read()
                    data = pickle.dumps(frame)
                    print("sending data...")
                    tosend = struct.pack("L", len(data))+data
                    conn.sendall(tosend)
            except:
                continue
            