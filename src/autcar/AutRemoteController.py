import socket
import json
import sys, signal

class RemoteController:

    def __init__(self, host = "", port = 8090):
        self.host = host
        self.port = port
        self.__connection = None
        self.__client_socket = None

    def connect(self):
        self.__connection = socket.socket()
        self.__connection.settimeout(10)
        self.__connection.connect((self.host, self.port))

    def listen(self):
        self.__connection = socket.socket()
        self.__connection.bind((self.host, self.port))
        self.__connection.listen(5)
        print("Listening on " + self.host + ":" + str(self.port))
  
    def send_cmd(self, cmd):
        message = {'cmd': cmd}
        self.__connection.send(json.dumps(message).encode())
        data = self.__connection.recv(4096)
        return data.decode()

    def close(self):
        if(self.__connection != None):
            self.__connection.close()
        if(self.__client_socket != None):
            self.__client_socket.close()

    def get_cmds(self):

        if(self.__client_socket == None):
            try:
                (client_socket, client_address) = self.__connection.accept()
                self.__client_socket = client_socket
            except KeyboardInterrupt:
                pass

        while True:
            try:
                client_input = b""
                client_input = self.__client_socket.recv(4096)
                if(len(client_input) > 0):
                    cmd = json.loads(client_input.decode())
                    retmsg = "received"
                    self.__client_socket.send(retmsg.encode())
                    return cmd['cmd']
            except KeyboardInterrupt:
                self.close()
                break