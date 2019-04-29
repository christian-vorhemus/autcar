from typing import List, TypeVar, Union
import ast
import csv
import os
from pathlib import Path
import numpy as np
import shutil
import random
import time
import xml.etree.cElementTree as et
import xml.dom.minidom
import cntk
from cntk.learners import learning_parameter_schedule
from cntk.ops import input_variable
from cntk.io import MinibatchSource, ImageDeserializer, StreamDefs, StreamDef
import cntk.io.transforms as xforms
from cntk.layers import default_options, Dense, Sequential, Activation, Embedding, Convolution2D, MaxPooling, Stabilizer, Convolution, Dropout, BatchNormalization
from cntk.ops.functions import CloneMethod
from cntk.logging import ProgressPrinter
from cntk.losses import cross_entropy_with_softmax
from cntk import classification_error, softmax, relu, ModelFormat, element_times, momentum_schedule, momentum_sgd

class Trainer:

    def __init__(self, image_width = 223, image_height = 168):
        """
        A trainer object that is used to prepare the dataset, train and test the model
        """
        self.__image_width = image_width
        self.__image_height = image_height

    def __saveMean(self, fname, image_width, image_height, data):
        root = et.Element('opencv_storage')
        et.SubElement(root, 'Channel').text = '3'
        et.SubElement(root, 'Row').text = str(image_height)
        et.SubElement(root, 'Col').text = str(image_width)
        meanImg = et.SubElement(root, 'MeanImg', type_id='opencv-matrix')
        et.SubElement(meanImg, 'rows').text = '1'
        et.SubElement(meanImg, 'cols').text = str(image_height * image_width * 3)
        et.SubElement(meanImg, 'dt').text = 'f'
        et.SubElement(meanImg, 'data').text = ' '.join(['%e\n' % n if (i+1)%4 == 0 else '%e' % n for i, n in enumerate(np.reshape(data, (image_height * image_width * 3)))])

        tree = et.ElementTree(root)
        tree.write(fname)
        x = xml.dom.minidom.parse(fname)
        with open(fname, 'w') as f:
            f.write(x.toprettyxml(indent = '  '))
            

    def create_balanced_dataset(self, path_to_folders: Union[str, List[str]], outputfolder_path = 'balanced_dataset', train_test_split = 0.8):
        image_counter = 0
        command_counter_start = {}
        command_counter = {}
        ignore_list = ["stop", "move_backwards"]
        commands = ["move_forward", "left_medium", "right_medium", "left_light", "right_light"]
        # Label encoding
        # 0 = move_forward
        # 1 = left_medium
        # 2 = right_medium
        # 3 = left_light
        # 4 = right_light

        if(type(path_to_folders) == str):
            files = [path_to_folders]
        elif(isinstance(path_to_folders, list)):
            files = path_to_folders

        for file in files:
            file = file.rstrip('/')
            try:
                with open(file+"/training.csv") as training_file:
                    csv_reader = csv.reader(training_file, delimiter=';')

                    for row in csv_reader:
                        try:
                            command = ast.literal_eval(row[1])
                        except:
                            continue
                        cmd_type = command["type"]
                        if(cmd_type == "move"):
                            cmd_type = command["type"] + "_" + command["direction"]
                        if(cmd_type == "left" or cmd_type == "right"):
                            cmd_type = command["type"] + "_" +command["style"]
                        if(cmd_type in ignore_list):
                            continue

                        if(command_counter_start.get(cmd_type, 0) == 0):
                            command_counter_start[cmd_type] = 1
                        else:
                            command_counter_start[cmd_type] = command_counter_start[cmd_type] + 1

                    training_file.seek(0)

            except Exception as e:
                raise Exception("No training.csv file found in folder "+file+". Does the directory path you provided resolve to a training folder created by AutCar?")

        max_class = max(command_counter_start, key = lambda x: command_counter_start.get(x))
        maximum = command_counter_start[max_class]

        outputfolder_path = outputfolder_path.rstrip('/')
        if not os.path.exists(outputfolder_path):
            os.makedirs(outputfolder_path)

        dataMean = np.full((3*self.__image_height*self.__image_width,), 128)
        self.__saveMean(outputfolder_path+"/meanfile.xml", self.__image_width, self.__image_height, dataMean)
        train_file = open(outputfolder_path+"/train_map.txt","w+")
        test_file = open(outputfolder_path+"/test_map.txt","w+")

        for file in files:
            file = file.rstrip('/')
            with open(file+"/training.csv") as training_file:
                csv_reader = csv.reader(training_file, delimiter=';')
                for row in csv_reader:
                    image = row[0]
                    try:
                        command = ast.literal_eval(row[1])
                    except:
                        continue
                    cmd_type = command["type"]
                    if(cmd_type == "move"):
                        cmd_type = command["type"] + "_" + command["direction"]
                    if(cmd_type == "left" or cmd_type == "right"):
                        cmd_type = command["type"] + "_" +command["style"]
                    if(cmd_type in ignore_list):
                        continue

                    if(command_counter.get(cmd_type, 0) == 0):
                        command_counter[cmd_type] = 1
                    else:
                        command_counter[cmd_type] = command_counter[cmd_type] + 1

                    shutil.copy(file+"/"+image, outputfolder_path+"/image_"+str(image_counter)+".png")
                    info = outputfolder_path+"/image_"+str(image_counter)+".png"+"\t"+str(commands.index(cmd_type))+"\n"
                    if(random.uniform(0, 1) < train_test_split):
                        train_file.write(info)
                    else:
                        test_file.write(info)
                    image_counter = image_counter + 1

                training_file.seek(0)

        finished = False
        while not finished:
            for file in files:
                if(finished == True):
                    break
                file = file.rstrip('/')
                with open(file+"/training.csv") as training_file:
                    csv_reader = csv.reader(training_file, delimiter=';')
                    for row in csv_reader:
                        image = row[0]
                        try:
                            command = ast.literal_eval(row[1])
                        except:
                            continue
                        cmd_type = command["type"]
                        if(cmd_type == "move"):
                            cmd_type = command["type"] + "_" + command["direction"]
                        if(cmd_type == "left" or cmd_type == "right"):
                            cmd_type = command["type"] + "_" +command["style"]
                        if(cmd_type in ignore_list):
                            continue

                        if(command_counter[cmd_type] < maximum):
                            command_counter[cmd_type] = command_counter[cmd_type] + 1
                            shutil.copy(file+"/"+image, outputfolder_path+"/image_"+str(image_counter)+".png")
                            info = outputfolder_path+"/image_"+str(image_counter)+".png"+"\t"+str(commands.index(cmd_type))+"\n"
                            if(random.uniform(0, 1) < train_test_split):
                                train_file.write(info)
                            else:
                                test_file.write(info)
                            image_counter = image_counter + 1

                    finished = True
                    for key, value in command_counter.items():
                        if(value < maximum):
                            finished = False
                            break

        train_file.close()
        test_file.close()

        return True

    def train(self, path_to_folder, epochs = 10, output_model_path = "driver_model.onnx"):

        path_to_folder = path_to_folder.rstrip('/')

        map_file_train = path_to_folder+"/train_map.txt"
        map_file_test = path_to_folder+"/test_map.txt"
        mean_file = path_to_folder+"/meanfile.xml"
        classes_set = set()
        num_classes = 3
        num_train = 0
        num_test = 0
        num_channels = 3

        try:
            with open(map_file_train) as f:
                csv_reader = csv.reader(f, delimiter='\t')
                for row in csv_reader:
                    cmd = row[1]
                    classes_set.add(cmd)
                    num_train = num_train + 1
        except Exception as e:
            raise Exception("No train_map.txt file found in path "+path_to_folder+". Did you create a dataset using create_balanced_dataset()?")

        num_classes = len(classes_set)

        with open(map_file_test) as f:
            for num_test, l in enumerate(f):
                pass

        transforms = [
            xforms.scale(width=self.__image_width, height=self.__image_height, channels=num_channels, interpolations='linear'),
            xforms.mean(mean_file),
        ]

        # ImageDeserializer loads images in the BGR format, not RGB
        reader_train = MinibatchSource(ImageDeserializer(map_file_train, StreamDefs(
            features = StreamDef(field='image', transforms=transforms),
            labels   = StreamDef(field='label', shape=num_classes)
        )))

        reader_test = MinibatchSource(ImageDeserializer(map_file_test, StreamDefs(
            features = StreamDef(field='image', transforms=transforms),
            labels   = StreamDef(field='label', shape=num_classes)
        )))

        input_var = input_variable((num_channels, self.__image_height, self.__image_width))
        label_var = input_variable((num_classes))

        # Normalize the input
        feature_scale = 1.0 / 256.0
        input_var_norm = element_times(feature_scale, input_var)

        create_model = Sequential([
            Convolution2D(filter_shape=(5,5), num_filters=32, strides=(1,1), pad=True, name="first_conv"),
            Activation(relu),
            MaxPooling(filter_shape=(3,3), strides=(2,2), name="first_max"),
            Convolution2D(filter_shape=(3,3), num_filters=48, strides=(1,1), pad=True, name="second_conv"),
            Activation(relu),
            MaxPooling(filter_shape=(3,3), strides=(2,2), name="second_max"),
            Convolution2D(filter_shape=(3,3), num_filters=64, strides=(1,1), pad=True, name="third_conv"),
            Activation(relu),
            MaxPooling(filter_shape=(3,3), strides=(2,2), name="third_max"),
            Convolution2D(filter_shape=(5,5), num_filters=32, strides=(1,1), pad=True, name="fifth_conv"),
            Activation(relu),
            Dense(100, activation=relu),
            Dropout(0.1),
            Dense(num_classes, activation=softmax)
        ])

        model = create_model(input_var)

        ce = cross_entropy_with_softmax(model, label_var)
        pe = classification_error(model, label_var)

        epoch_size = num_train
        minibatch_size = 64

        lr_per_minibatch = learning_parameter_schedule([0.01]*10 + [0.003]*10 + [0.001], epoch_size = epoch_size)
        momentums = momentum_schedule(0.9, minibatch_size = minibatch_size)
        l2_reg_weight = 0.001

        learner = momentum_sgd(model.parameters, lr = lr_per_minibatch, momentum = momentums, l2_regularization_weight=l2_reg_weight)
        progress_printer = ProgressPrinter(tag='Training', num_epochs=epochs)
        trainer = cntk.train.Trainer(model, (ce, pe), [learner], [progress_printer])

        input_map = {
            input_var: reader_train.streams.features,
            label_var: reader_train.streams.labels
        }

        batch_index = 0
        plot_data = {'batchindex':[], 'loss':[], 'error':[]}
        for epoch in range(epochs):
            sample_count = 0
            while sample_count < epoch_size:
                data = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count), input_map=input_map)

                trainer.train_minibatch(data)
                sample_count += data[label_var].num_samples

                plot_data['batchindex'].append(batch_index)
                plot_data['loss'].append(trainer.previous_minibatch_loss_average)
                plot_data['error'].append(trainer.previous_minibatch_evaluation_average)

                batch_index += 1
            trainer.summarize_training_progress()

        epoch_size = num_test
        minibatch_size = 16

        metric_numer = 0
        metric_denom = 0
        sample_count = 0
        minibatch_index = 0

        while sample_count < epoch_size:
            current_minibatch = min(minibatch_size, epoch_size - sample_count)

            data = reader_test.next_minibatch(current_minibatch, input_map=input_map)

            metric_numer += trainer.test_minibatch(data) * current_minibatch
            metric_denom += current_minibatch

            sample_count += data[label_var].num_samples
            minibatch_index += 1

        print("")
        print("Final Results: Minibatch[1-{}]: errs = {:0.1f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom))
        print("")

        model.save(output_model_path, format=ModelFormat.ONNX)


    def test(self, path_to_model, path_to_test_map):
        print("test")