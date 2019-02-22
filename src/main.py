from autcar import Camera, Car, RemoteController, Capture

cam = Camera()
rc = RemoteController()
car = Car()
cap = Capture(car, cam)

rc.listen()

def live_camera():
    print("Camera started")
    cam.listen()

#Thread(target=live_camera).start()

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
        cap.start()
    if(cmd == "stoprecording"):
        cap.stop()
        
