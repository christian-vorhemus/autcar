from autcar import Camera, Car, RemoteController
from threading import Thread
import time

cam = Camera()
rc = RemoteController()
rc.listen()
car = Car()

def live_camera():
    print("Camera started")
    cam.listen()

counter = 1
stoprecord = False
def record_data(car):
    global counter
    global stoprecord
    while True:
        if(stoprecord):
            break
        carcmd = car.current_command

        print(carcmd)
        cmd = ""

        if(carcmd['type'] == "move"):
            if(carcmd['speed'] == "medium"):
                cmd = "fast"
            if(carcmd['speed'] == "fast"):
                cmd = "faster"
        if(carcmd['type'] == "stop"):
            cmd = "stop"
        if(carcmd['type'] == "left"):
            if(carcmd['style'] == "light"):
                cmd = "leftlight"
            if(carcmd['style'] == "medium"):
                cmd = "lefthard"
        if(carcmd['type'] == "right"):
            if(carcmd['style'] == "light"):
                cmd = "rightlight"
            if(carcmd['style'] == "medium"):
                cmd = "righthard"

        cam.take_snapshot(counter)
        with open('autcar_train/training.csv', 'a') as f:
            f.write(str(counter) + "_car_snapshot.png;" + cmd + "\r\n")
        counter = counter + 1
        time.sleep(3)

Thread(target=live_camera).start()

proc = Thread(target=record_data, args=(car,))
direction = None

while True:
    cmd = rc.get_cmds()
    print(cmd)
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
        proc.start()
    if(cmd == "stoprecording"):
        stoprecord = True
        
