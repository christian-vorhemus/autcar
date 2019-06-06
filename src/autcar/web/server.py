from flask import Flask, render_template, Response, request, jsonify
import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AutCamera import Camera
from AutRemoteController import RemoteController
 
app = Flask(__name__)
rc = None
car_ip = ""
camera_port = 0
car_port = 0
connected = False

def gen():
    global car_ip
    global camera_port
    global car_port
    global connected
    camera = None
    while True:
        if(camera_port != 0):
            if(connected == False):
                camera = Camera(connect_camera=True, host=car_ip, port=camera_port)
                print("Camera object created on "+car_ip+":"+str(camera_port))
                connected = True

            frame = camera.get_frame()

            if(frame is None):
                time.sleep(1)
                continue
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            time.sleep(1)

@app.route('/')
def home():
    try:
        return render_template("index.html")
    except Exception as e:
        return str(e)

@app.route('/disconnect')
def disconnect():
    rc.close()
    return jsonify({'status': 'success', 'type': 'disconnected'})

@app.route('/connect')
def connect():
    global rc
    global car_ip
    global car_port
    global camera_port
    global connected

    address = request.args.get('address')

    if(":" in address):
        car_ip, car_port = address.split(":")
        camera_port = int(car_port)-1
        car_port = int(car_port)
    else:
        car_ip = address
        camera_port = 8089
        car_port = 8090

    try:
        rc = RemoteController(car_ip, car_port)
        time.sleep(1)
        rc.connect()
        message = {'status': 'success', 'type': 'connected'}
    except ConnectionRefusedError:
        if(connected == False):
            message = {'status': 'error', 'type': 'connection_refused'}
        else:
            message = {'status': 'success', 'type': 'camera_connected'}

    return jsonify(message)


@app.route('/cmds')
def cmds():

    global rc
    cmd = request.args.get('cmd')
    print("Sent command: " + cmd)

    message = {}

    if(cmd == "fast"):
        rc.send_cmd("fast")
    if(cmd == "faster"):
        rc.send_cmd("faster")
    if(cmd == "backwards"):
        rc.send_cmd("backwards")
    if(cmd == "leftlight"):
        rc.send_cmd("leftlight")
    if(cmd == "lefthard"):
        rc.send_cmd("lefthard")
    if(cmd == "rightlight"):
        rc.send_cmd("rightlight")
    if(cmd == "righthard"):
        rc.send_cmd("righthard")
    if(cmd == "stop"):
        rc.send_cmd("stop")
    if(cmd == "startrecording"):
        try:
            rc.send_cmd("startrecording")
            message = {'status': 'success', 'type': 'recording_started'}
        except:
            message = {'status': 'error', 'type': 'recording_notstarted'}
    if(cmd == "stoprecording"):
        try:
            rc.send_cmd("stoprecording")
            message = {'status': 'success', 'type': 'recording_stopped'}
        except:
            message = {'status': 'error', 'type': 'recording_notstopped'}
    if(cmd == "retrievedata"):
        message = {'status': 'error', 'type': 'not_implemented'}
    if(cmd == "setmlmodel"):
        message = {'status': 'error', 'type': 'not_implemented'}

    return jsonify(message)

@app.route('/video')
def video():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

 
if __name__ == "__main__":
	app.run(debug = True, host='0.0.0.0', port=8080, passthrough_errors=True)