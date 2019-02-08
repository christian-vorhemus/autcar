from flask import Flask, render_template, Response, request, jsonify
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AutCamera import Camera
from AutRemoteController import RemoteController
 
app = Flask(__name__)
rc = RemoteController("192.168.1.121")


def gen(camera):
    while True:
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

@app.route('/cmds')
def cmds():

    cmd = request.args.get('cmd')
    print("Sent command: " + cmd)

    message = {'status': 'success', 'type': 'cmd_accepted'}
    if(cmd == "connect"):
        try:
            rc.connect()
            message = {'status': 'success', 'type': 'connected'}
        except ConnectionRefusedError:
            message = {'status': 'error', 'type': 'connection_refused'}
    if(cmd == "fast"):
        rc.send_cmd("fast")
    if(cmd == "faster"):
        rc.send_cmd("faster")
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
        rc.send_cmd("startrecording")
    if(cmd == "stoprecording"):
        rc.send_cmd("startrecording")
    if(cmd == "retrievedata"):
        message = {'status': 'error', 'type': 'not_implemented'}
    if(cmd == "setmlmodel"):
        message = {'status': 'error', 'type': 'not_implemented'}

    return jsonify(message)

@app.route('/video')
def video():
    return Response(gen(Camera(True, "192.168.1.121")), mimetype='multipart/x-mixed-replace; boundary=frame')

 
if __name__ == "__main__":
	app.run(debug = True, host='0.0.0.0', port=8080, passthrough_errors=True)