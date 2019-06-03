from threading import Thread
from typing import List
from multiprocessing import Process, Value
import time
import onnxruntime as rt
import cv2
import os
from PIL.ImageOps import equalize
from PIL import Image
import numpy as np
from autcar import Camera, Car

class Model:
    def __init__(self, model_file_path: str):
        self.model_file_path = model_file_path
        self.last_command = None
    def preprocess(self, image: Image):
        """
        Preprocess is used to adjust (e.g. scale) the image before it is handed over to the neural network for prediction. It must return a numpy array representation of the image: [1 x channels x image_height x image_width]

        @param image: You get a Pillow Image object from the car you can use for scaling, normalization etc
        """
        processed_image = image.resize((224,168), Image.LINEAR)
        X = np.array([np.moveaxis((np.array(processed_image).astype('float32')), -1, 0)])
        X -= np.mean(X, keepdims=True)
        X /= (np.std(X, keepdims=True) + 1e-6)
        return X
    def execute(self, prediction: int, car: Car):
        """
        Tells the car what to do after the prediction was made.

        @param prediction: The integer (index) prediction value of the model
        @param car: The car object you can use to perform actions
        """
        if(prediction == 0):
            if(self.last_command == "left_light_backwards"):
                return
            self.last_command = "left_light_backwards"
            print(self.last_command)
            car.left("light", "backwards")
        elif(prediction == 1):
            if(self.last_command == "left_light_forward"):
                return
            self.last_command = "left_light_forward"
            print(self.last_command)
            car.left("light", "forward")
        elif(prediction == 2):
            if(self.last_command == "left_medium_forward"):
                return
            self.last_command = "left_medium_forward"
            print(self.last_command)
            car.left("medium", "forward")
        elif(prediction == 3):
            if(self.last_command == "left_medium_backwards"):
                return
            self.last_command = "left_medium_backwards"
            print(self.last_command)
            car.left("medium", "backwards")
        elif(prediction == 4):
            if(self.last_command == "move_fast_forward"):
                return
            self.last_command = "move_fast_forward"
            print(self.last_command)
            car.move("forward", "fast")
        elif(prediction == 5):
            if(self.last_command == "move_medium_backwards"):
                return
            self.last_command = "move_medium_backwards"
            print(self.last_command)
            car.move("backwards", "medium")
        elif(prediction == 6):
            if(self.last_command == "move_medium_forward"):
                return
            self.last_command = "move_medium_forward"
            print(self.last_command)
            car.move("forward", "medium")
        elif(prediction == 7):
            if(self.last_command == "right_light_backwards"):
                return
            self.last_command = "right_light_backwards"
            print(self.last_command)
            car.right("light", "backwards")
        elif(prediction == 8):
            if(self.last_command == "right_light_forward"):
                return
            self.last_command = "right_light_forward"
            print(self.last_command)
            car.right("light", "forward")
        elif(prediction == 9):
            if(self.last_command == "right_medium_backwards"):
                return
            self.last_command = "right_medium_backwards"
            print(self.last_command)
            car.right("medium", "backwards")
        elif(prediction == 10):
            if(self.last_command == "right_medium_forward"):
                return
            self.last_command = "right_medium_forward"
            print(self.last_command)
            car.right("medium", "forward")
        elif(prediction == 11):
            if(self.last_command == "stop"):
                return
            self.last_command = "stop"
            print(self.last_command)
            car.stop()


class Driver:

    def __init__(self, model_instance_list: List[Model], car: Car, camera: Camera, execution_interval: int = 2):
        """
        A Driver object is used to autonomously drive a car. It needs a car object and a path to a model file

        @param model_instance_list: A list containing models that point to the path of .onnx model location
        @param car: The car object which is used to control the motor
        @param camera: A camera object used to let the car take pictures from the environment
        @param execution_interval: Defines how often the model is executed. Default is 2 seconds
        """
        self.__car = car
        self.__cam = camera

        threads = []
        for model_instance in model_instance_list:
            if(os.path.isfile(model_instance.model_file_path) == False):
                raise Exception("Error: File %s does not exist. Did you train and create a model file?"%model_instance.model_file_path)
                return
            thread = Thread(target=self.__drive_onnx, args=(model_instance,))
            threads.append(thread)

        self.__model_threads = threads
        self.__frame = None
        self.__stop_driving = False
        self.__capture_interval = execution_interval
        self.__counter = 0
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

    def __normalize(self, arr, desired_mean = 0, desired_std = 1):
        arr = arr.astype('float')
        for i in range(3):
            mean = arr[...,i].mean()
            std = arr[...,i].std()
            arr[...,i] = (arr[...,i] - mean)*(desired_std/std) + desired_mean
        return arr

    def __scale_image(self, image, scaling=(224,168)):
        try:
            return image.resize(scaling, Image.LINEAR)
        except:
            raise Exception('pad_image error')

    def __drive_keras(self):
        from keras.models import load_model
        self.__last_timestamp = time.time()

        try:
            model = load_model(self.__model_file)
        except Exception as e:
            print("Model file not available")

        while True:
            if(self.__stop_driving):
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
        import tensorflow as tf
        from tensorflow.python.platform import gfile
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
                if(self.__stop_driving):
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


    def __drive_onnx(self, model_instance: Model):
        
        self.__last_timestamp = time.time()
        sess = rt.InferenceSession(model_instance.model_file_path)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name

        while True:
            if(self.__stop_driving):
                break

            # We constantly read new images from the cam to empty the VideoCapture buffer
            ret, frame = self.__cam.read()
            self.__frame = frame
            current_time = time.time()

            if(current_time - self.__last_timestamp > self.__capture_interval):
                self.__last_timestamp = current_time
                try:
                    # OpenCV reads BGR, Pillow reads RGB -> convert
                    imgconv = cv2.cvtColor(self.__frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(imgconv)
                except Exception as e:
                    print("Cant read image")
                try:
                    X = model_instance.preprocess(img)
                except Exception as e:
                    print("Error while preprocessing image: "+e)

                #r, g, b = processed_image.split()
                #processed_image = Image.merge("RGB", (b, g, r))
                #X = np.array([np.moveaxis((np.array(processed_image).astype('float32')-128), -1, 0)])

                pred = sess.run([label_name], {input_name: X.astype(np.float32)})[0]
                prediction = int(np.argmax(pred))

                model_instance.execute(prediction, self.__car)


    # def __drive_onnx_new(self):
    #     self.__last_timestamp = time.time()
    #     sess = rt.InferenceSession(self.__model_file)
    #     input_name = sess.get_inputs()[0].name
    #     label_name = sess.get_outputs()[0].name
    #     index = Value("i", -1)

    #     def norm(_img):
    #         tt=np.asarray(_img).astype('float32')
    #         tt=tt/255
    #         tt[0]=(tt[0]-0.485)/0.229
    #         tt[1]=(tt[1]-0.456)/0.224
    #         tt[2]=(tt[2]-0.406)/0.225
    #         return tt

    #     while True:
    #         if(self.__stop_driving):
    #             break

    #         ret, frame = self.__cam.read()
    #         self.__frame = frame
    #         current_time = time.time()

    #         if(current_time - self.__last_timestamp > self.__capture_interval):
    #             self.__last_timestamp = current_time
    #             try:
    #                 imgconv = cv2.cvtColor(self.__frame, cv2.COLOR_BGR2RGB)
    #                 img = Image.fromarray(imgconv)
    #             except Exception as e:
    #                 print("Cant read image")
    #             try:
    #                 processed_image = norm(self.__scale_image(img, (224, 168)))
    #             except Exception as e:
    #                 print(e)
    #                 print("Error while reading image")

    #             X = np.array([np.moveaxis(processed_image, -1, 0)])

    #             pred = sess.run([label_name], {input_name: X.astype(np.float32)})[0]
    #             index = np.argmax(pred)

    #             if(index == 0):
    #                 if(self.__last_command == "forward"):
    #                     continue
    #                 print("forward")
    #                 self.__last_command = "forward"
    #                 self.__car.move("forward", "medium")
    #             elif(index == 1):
    #                 if(self.__last_command == "left"):
    #                     continue
    #                 print("left")
    #                 self.__last_command = "left"
    #                 self.__car.left("light", "forward")
    #             elif(index == 2):
    #                 if(self.__last_command == "right"):
    #                     continue
    #                 print("right")
    #                 self.__last_command = "right"
    #                 self.__car.right("light", "forward")

    
    def start(self):
        """
        Start the self driving methods
        """
        print("Driver started")
        try:
            for model_thread in self.__model_threads:
                model_thread.start()
        except KeyboardInterrupt:
            print("Keyboard interrupt")
            exit()

    def stop(self):
        self.__stop_driving = True