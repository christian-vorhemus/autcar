from flask import Flask, render_template, Response, request, jsonify
import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AutCamera import Camera
from AutRemoteController import RemoteController
 
app = Flask(__name__)
rc = None
car_ip = ""
car_port = 0
connected = False

def gen():
    global car_ip
    global car_port
    global connected
    camera = None
    while True:
        if(car_port != 0):
            if(connected == False):
                camera = Camera(True, car_ip, car_port-1)
                print("Camera object created on "+car_ip+":"+str(car_port-1))
                connected = True

            frame = camera.get_frame()
            if(frame is None):
                continue
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

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

    address = request.args.get('address')

    car_ip, car_port = address.split(":")
    car_port = int(car_port)

    try:
        rc = RemoteController(car_ip, car_port)
        rc.connect()
        message = {'status': 'success', 'type': 'connected'}
    except ConnectionRefusedError:
        message = {'status': 'error', 'type': 'connection_refused'}

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