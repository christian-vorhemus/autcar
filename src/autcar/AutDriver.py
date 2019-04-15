from threading import Thread
import time
import onnxruntime as rt
import cv2
import os
from PIL.ImageOps import equalize
from PIL import Image
import numpy as np
from AutCamera import Camera
#from keras.models import load_model
#import tensorflow as tf
#from tensorflow.python.platform import gfile

class Driver:

    def __init__(self, model, car, capture_interval = 2, rotation = -1):
        self.__car = car
        self.__cam = Camera(rotation=rotation)
        #self.__cam = cv2.VideoCapture(0)

        if(os.path.isfile(model) == False):
            raise Exception("Error: File %s does not exist. Did you train and create a model file?"%model)
            return

        self.__model_file = model
        self.__frame = None
        self.__proc = Thread(target=self.__drive_onnx)
        self.__stop_recording = False
        self.__capture_interval = capture_interval
        self.__counter = 0
        self.__last_command = None
        self.__last_timestamp = 0

    def __pad_image(self, image):
        target_size = max(image.size)
        result = Image.new('RGB', (target_size, target_size), "white")
        try:
            result.paste(image, (int((target_size - image.size[0]) / 2), int((target_size - image.size[1]) / 2)))
        except:
            print("Error on image " + image)
            raise Exception('pad_image error')
        return result

    def __scale_image(self, image):
        try:
            return image.resize((128,128))
        except:
            raise Exception('pad_image error')

    def __drive_keras(self):
        self.__last_timestamp = time.time()

        try:
            model = load_model(self.__model_file)
        except Exception as e:
            print("Model file not available")

        while True:
            if(self.__stop_recording):
                break

            # We constantly read new images from the cam to empty the VideoCapture buffer
            ret, frame = self.__cam.read()
            self.__frame = frame
            current_time = time.time()

            if(current_time - self.__last_timestamp > self.__capture_interval):
                self.__last_timestamp = current_time

                try:
                    img = Image.fromarray(self.__frame)
                except Exception as e:
                    print("Cant read image")
                try:
                    processed_image = equalize(self.__scale_image(self.__pad_image(img)))
                except Exception as e:
                    print("Err while reading image")

                X = np.array(processed_image)/255.0
                X = np.expand_dims(X, axis=0)

                pred = model.predict(X)
                index = np.argmax(pred)

                if(index == 0):
                    if(self.__last_command == "forward"):
                        continue
                    print("forward")
                    self.__last_command = "forward"
                    self.__car.move("forward", "medium")
                elif(index == 1):
                    if(self.__last_command == "left"):
                        continue
                    print("left")
                    self.__last_command = "left"
                    self.__car.left("light", "forward")
                elif(index == 2):
                    if(self.__last_command == "right"):
                        continue
                    print("right")
                    self.__last_command = "right"
                    self.__car.right("light", "forward")


    def __drive_tensorflow(self):
        self.__last_timestamp = time.time()

        with tf.Session() as sess:
            print("load graph")
            with gfile.FastGFile(self.__model_file, 'rb') as f:
                graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            graph_nodes=[n for n in graph_def.node]

            softmax_tensor = sess.graph.get_tensor_by_name('Softmax612:0')

            while True:
                if(self.__stop_recording):
                    break

                # We constantly read new images from the cam to empty the VideoCapture buffer
                ret, frame = self.__cam.read()
                self.__frame = frame
                current_time = time.time()

                if(current_time - self.__last_timestamp > self.__capture_interval):
                    self.__last_timestamp = current_time

                    try:
                        img = Image.fromarray(self.__frame)
                    except Exception as e:
                        print("Cant read image")
                    try:
                        processed_image = equalize(self.__scale_image(self.__pad_image(img)))
                    except Exception as e:
                        print("Err while reading image")

                    X = np.array([np.moveaxis(np.array(processed_image), -1, 0)])/255.0

                    pred = sess.run(softmax_tensor, {'Input501:0': X})
                    print(pred)

                    index = np.argmax(pred)

                    if(index == 0):
                        if(self.__last_command == "forward"):
                            continue
                        print("forward")
                        self.__last_command = "forward"
                        self.__car.move("forward", "medium")
                    elif(index == 1):
                        if(self.__last_command == "left"):
                            continue
                        print("left")
                        self.__last_command = "left"
                        self.__car.left("light", "forward")
                    elif(index == 2):
                        if(self.__last_command == "right"):
                            continue
                        print("right")
                        self.__last_command = "right"
                        self.__car.right("light", "forward")


    def __drive_onnx(self):
        self.__last_timestamp = time.time()

        while True:
            if(self.__stop_recording):
                break

            # We constantly read new images from the cam to empty the VideoCapture buffer
            ret, frame = self.__cam.read()
            self.__frame = frame
            current_time = time.time()

            if(current_time - self.__last_timestamp > self.__capture_interval):
                self.__last_timestamp = current_time
                try:
                    img = Image.fromarray(self.__frame)
                except Exception as e:
                    print("Cant read image")
                try:
                    processed_image = equalize(self.__scale_image(self.__pad_image(img)))
                except:
                    print("Err while reading image")

                X = np.array([np.moveaxis(np.array(processed_image), -1, 0)])/255.0

                sess = rt.InferenceSession(self.__model_file)
                input_name = sess.get_inputs()[0].name
                label_name = sess.get_outputs()[0].name
                pred = sess.run([label_name], {input_name: X.astype(np.float32)})[0]
                index = np.argmax(pred)
                index = 0

                if(index == 0):
                    if(self.__last_command == "forward"):
                        continue
                    print("forward")
                    self.__last_command = "forward"
                    self.__car.move("forward", "medium")
                elif(index == 1):
                    if(self.__last_command == "left"):
                        continue
                    print("left")
                    self.__last_command = "left"
                    self.__car.move("leftlight", "forward")
                elif(index == 2):
                    if(self.__last_command == "right"):
                        continue
                    print("right")
                    self.__last_command = "right"
                    self.__car.move("rightlight", "forward")

    def start(self):
        print("Auto driver started")
        self.__proc.start()

    def stop(self):
        self.__stop_recording = True