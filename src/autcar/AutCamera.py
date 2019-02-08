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
        #wd = os.getcwd()
        #print(wd)
        #os.chdir(wd+"")
        script_dir = os.path.dirname(__file__)
        rel_path = "2091/data.txt"
        abs_file_path = os.path.join(script_dir, rel_path)
        self.__nosignal = True
        if(capture):
            self.__nosignalframes = [open('autcar/web/res/' + f + '.jpg', 'rb').read() for f in ['f1', 'f2', 'f3']]
            threading.Thread(target=self.frame_updater).start()


    def take_snapshot(self, counter = None):
        ret, frame = self.__cam.read()
        if(counter is None):
            counter = self.__counter
            self.__counter = counter + 1
        if not os.path.exists("autcar_train"):
            os.mkdir("autcar_train")
        img_name = "autcar_train/{}_car_snapshot.png".format(counter)
        cv2.imwrite(img_name, frame)


    def get_frame(self):
        if(self.__nosignal == False):
            return self.__frame
        else:
            fr = self.__nosignalframes[int(time.time()) % 3]
            return fr


    def frame_updater(self):
        print("Called")
        clientsocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        try:
            clientsocket.connect((self.host, self.port))
        except:
            self.__nosignal = True
            return

        print("Socket started")

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
            return jpeg.tobytes()
            #cv2.imshow('frame', frame)
            #cv2.waitKey(1)


    def listen(self, host = '', port = 8089):

        serversocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        serversocket.bind((host, port))
        serversocket.listen(10)
        print('Socket now listening on ' + host + ":" + str(port))

        while True:
            try:
                print("New client connection")
                conn, addr = serversocket.accept()
                while True:
                    ret, frame = self.__cam.read()
                    data = pickle.dumps(frame)
                    tosend = struct.pack("L", len(data))+data
                    conn.sendall(tosend)
            except:
                continue
            