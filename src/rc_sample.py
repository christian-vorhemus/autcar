from autcar import Camera, Car, RemoteController, Capture
from threading import Thread
import time

rc = RemoteController()
car = Car()
cam = Camera(rotation=-1)
cap = Capture(car, cam, capture_interval=2)

rc.listen()
direction = None

while True:
    cmd = rc.get_cmds()
    print(cmd + ", time:" + str(int(time.time())))
    if(cmd == "fast"):
        direction = "forward"
        car.move("forward", "medium")
    if(cmd == "stop"):
        car.stop()
    if(cmd == "faster"):
        direction = "forward"
        car.move("forward", "fast")
    if(cmd == "backwards"):
        direction = "backwards"
        car.move("backwards")
    if(cmd == "leftlight"):
        car.left("light", direction)
    if(cmd == "lefthard"):
        car.left("medium", direction)
    if(cmd == "rightlight"):
        car.right("light", direction)
    if(cmd == "righthard"):
        car.right("medium", direction)
    if(cmd == "startrecording"):
        cap.start()
    if(cmd == "stoprecording"):
        cap.stop()
        
