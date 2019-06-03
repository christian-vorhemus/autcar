import socket
import json
import sys, signal

class RemoteController:

    def __init__(self, host = "", port = 8090):
        """
        An object that can either be used to create a listener for remote connections or connect to a listener.
        Remote connections are used to send and receive commands for the car

        @param host: If the object is used for sending data, host is the IP address of the car.
        @param port: The port the listener is binding on. Default is 8090
        """
        self.host = host
        self.port = port
        self.__connection = None
        self.__client_socket = None

    def connect(self):
        """
        Connects to a remote device. Is using IP address and port passed in the constructor. Should be called on the car
        """
        self.__connection = socket.socket()
        self.__connection.settimeout(10)
        self.__connection.connect((self.host, self.port))

    def listen(self):
        """
        Opens the possibility to connect. Is using IP address and port passed in the constructor. Should be called on the car
        """
        self.__connection = socket.socket()
        self.__connection.bind((self.host, self.port))
        self.__connection.listen(5)
        print("Listening on " + self.host + ":" + str(self.port))
  
    def send_cmd(self, cmd):
        """
        A method to send data to another device. Should be used on the device that talks to the car
        """
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
        """
        Method to receive commands send from another device. Should be called on the car
        """
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