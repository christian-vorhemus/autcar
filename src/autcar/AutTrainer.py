from typing import List, Union
import ast
import csv
import os
import math
import numpy as np
import shutil
import random
from PIL import Image
import onnxruntime as rt
import matplotlib.pyplot as plt

class Trainer:

    def __init__(self, deeplearning_framework = "keras", image_width = 224, image_height = 168):
        """
        A trainer object that is used to prepare the dataset, train and test the model
        """
        self.__image_width = image_width
        self.__image_height = image_height
        self.__deeplearning_framework = deeplearning_framework
        self.__commands = ['left_light_backwards', 'left_light_forward', 'left_medium_backwards', 'left_medium_forward', 'move_fast_forward', 'move_medium_backwards', 'move_medium_forward', 'right_light_backwards', 'right_light_forward', 'right_medium_backwards', 'right_medium_forward', 'stop']


    def __scale_image(self, image, width=None, height=None):

        if(width is None):
            width = self.__image_width
        if(height is None):
            height = self.__image_height

        try:
            return image.resize((width,height), Image.LINEAR)
        except:
            raise Exception('scale_image error')   

    def create_balanced_dataset(self, input_folder_path: Union[str, List[str]], output_folder_path: str = 'balanced_dataset', train_test_split: float = 0.8):
        image_counter = 0
        command_counter_start = {}
        command_counter = {}
        ignore_list = ["stop"]

        outputfolder_path = output_folder_path.rstrip('/')
        if os.path.exists(outputfolder_path):
            shutil.rmtree(outputfolder_path)
        
        os.makedirs(outputfolder_path)

        if(type(input_folder_path) == str):
            files = [input_folder_path]
        elif(isinstance(input_folder_path, list)):
            files = input_folder_path

        for file in files:
            file = file.rstrip('/')
            try:
                with open(file+"/training.csv") as training_file:
                    csv_reader = csv.reader(training_file, delimiter=';')

                    for row in csv_reader:
                        try:
                            command = ast.literal_eval(row[1])
                        except Exception as e:
                            continue
                        cmd_type = command["type"]
                        if(cmd_type == "move"):
                            cmd_type = command["type"] + "_" + command["speed"] + "_" + command["direction"]
                        if(cmd_type == "left" or cmd_type == "right"):
                            cmd_type = command["type"] + "_" +command["style"] + "_" + command["direction"]
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
                    except Exception as e:
                        continue
                    cmd_type = command["type"]
                    if(cmd_type == "move"):
                        cmd_type = command["type"] + "_" + command["speed"] + "_" + command["direction"]
                    if(cmd_type == "left" or cmd_type == "right"):
                        cmd_type = command["type"] + "_" +command["style"] + "_" + command["direction"]
                    if(cmd_type in ignore_list):
                        continue

                    if(command_counter.get(cmd_type, 0) == 0):
                        command_counter[cmd_type] = 1
                    else:
                        command_counter[cmd_type] = command_counter[cmd_type] + 1

                    shutil.copy(file+"/"+image, outputfolder_path+"/image_"+str(image_counter)+".png")
                    info = outputfolder_path+"/image_"+str(image_counter)+".png"+"\t"+str(cmd_type)+"\n"
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
                        except Exception as e:
                            continue
                        cmd_type = command["type"]
                        if(cmd_type == "move"):
                            cmd_type = command["type"] + "_" + command["speed"] + "_" + command["direction"]
                        if(cmd_type == "left" or cmd_type == "right"):
                            cmd_type = command["type"] + "_" +command["style"] + "_" + command["direction"]
                        if(cmd_type in ignore_list):
                            continue

                        if(command_counter[cmd_type] < maximum):
                            command_counter[cmd_type] = command_counter[cmd_type] + 1
                            shutil.copy(file+"/"+image, outputfolder_path+"/image_"+str(image_counter)+".png")
                            info = outputfolder_path+"/image_"+str(image_counter)+".png"+"\t"+str(cmd_type)+"\n"
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


    def get_classes(self, path_to_folder: str):
        path_to_folder = path_to_folder.rstrip('/')
        classes_dict = dict()
        map_file_train = path_to_folder+"/train_map.txt"

        try:
            with open(map_file_train) as f:
                csv_reader = csv.reader(f, delimiter='\t')
                for row in csv_reader:
                    cmd = row[1]
                    if(cmd in classes_dict):
                        classes_dict[cmd] += 1
                    else:
                        classes_dict[cmd] = 1

        except Exception as e:
            raise Exception("No train_map.txt file found in path "+path_to_folder+". Did you create a dataset using create_balanced_dataset()?")

        return classes_dict


    def train(self, path_to_folder: str, model_definition, epochs: int = 10, output_model_path: str = "driver_model.onnx", classes = None, minibatch_size: int = 64):
        if(classes is None):
            classes = self.__commands
        
        if(self.__deeplearning_framework == "cntk"):
            self.__train_cntk(path_to_folder, model_definition, epochs, output_model_path, classes, minibatch_size)
        elif(self.__deeplearning_framework == "pytorch"):
            self.__train_pytorch(path_to_folder, model_definition, epochs, output_model_path, classes, minibatch_size)
        else:
            # Use Keras as default
            self.__train_keras(path_to_folder, model_definition, epochs, output_model_path, classes, minibatch_size)


    def __train_keras(self, path_to_folder: str, model_definition, epochs: int, output_model_path: str, classes, minibatch_size: int):
        
        from keras.preprocessing.image import ImageDataGenerator
        from keras import Model
        import onnxmltools
        import pandas as pd

        model: Model = model_definition

        path_to_folder = path_to_folder.rstrip('/')
        map_file_train = path_to_folder+"/train_map.txt"
        map_file_test = path_to_folder+"/test_map.txt"
        minibatch_size = 16

        df_train = pd.read_csv(map_file_train, sep='\t', dtype=str, header=None, names=["filename", "class"])
        df_test = pd.read_csv(map_file_test, sep='\t', dtype=str, header=None, names=["filename", "class"])

        df_train_len = len(df_train)
        df_test_len = len(df_test)

        train_datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, data_format="channels_first")
        test_datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, data_format="channels_first")

        generator_train = train_datagen.flow_from_dataframe(df_train, classes=classes, target_size=(self.__image_height, self.__image_width), shuffle=True)
        generator_test = test_datagen.flow_from_dataframe(df_test, classes=classes, target_size=(self.__image_height, self.__image_width), shuffle=True)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print("Training started")
        res = model.fit_generator(generator_train, epochs=epochs, steps_per_epoch=round(df_train_len/minibatch_size), validation_data=generator_test, validation_steps=round(df_test_len/minibatch_size))

        validation_loss = res.history["val_loss"]
        validation_accuracy = res.history["val_acc"]

        onnx_model = onnxmltools.convert_keras(model) 
        onnxmltools.utils.save_model(onnx_model, output_model_path)

        return validation_loss, validation_accuracy


    def __train_pytorch(self, path_to_folder: str, model_definition, epochs: int, output_model_path: str, classes, minibatch_size: int):
        import torch
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader, Dataset
        from torch.nn import CrossEntropyLoss, Module
        from torch.optim import Adam
        from torchvision.datasets import MNIST
        import pandas as pd

        minibatch_size = 16
        num_channels = 3
        img_height = self.__image_height
        img_width = self.__image_width
        calculate_accuracy_every_iteration = False

        model: Module = model_definition

        class TrackDataset(Dataset):

            def __init__(self, path_to_mapfile, img_height, img_width):
                self.dataframes = pd.read_csv(path_to_mapfile, sep='\t', dtype=str, header=None, names=["filename", "class"])
                self.transform = transforms.Compose([
                    transforms.Resize((img_height,img_width)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])

            def __len__(self):
                return len(self.dataframes)

            def __getitem__(self, index):
                img_name = self.dataframes.iloc[index, 0]
                image = Image.open(img_name)
                cl = self.dataframes.iloc[index, 1:].values[0]
                image = self.transform(image)
                label = classes.index(cl)
                return image, label

        map_file_train = path_to_folder+"/train_map.txt"
        map_file_test = path_to_folder+"/test_map.txt"

        train_dataset = TrackDataset(map_file_train, img_height, img_width)
        test_dataset = TrackDataset(map_file_test, img_height, img_width)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=minibatch_size, shuffle=True)

        criterion = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=0.001)

        print("Training started")
        training_length = len(train_loader)
        for epoch in range(epochs):  
            losses = []
            accuracies = []
            for batch_index, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                curr_loss = loss.data.item()
                losses.append(curr_loss)

                if(calculate_accuracy_every_iteration == True):
                    correct = 0
                    total = 0
                    for test_images, test_labels in test_loader:
                        outputs = model(test_images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += test_labels.size(0)
                        correct += (predicted == test_labels).sum()
                        correct = correct.data.item()

                    curr_acc = correct/total
                    accuracies.append(curr_acc)

                if(batch_index % 100 == 0 and calculate_accuracy_every_iteration == True):
                    print('Iteration %d/%d - loss: %.4f - acc: %.4f' %(batch_index, training_length, curr_loss, curr_acc))
                if(batch_index % 100 == 0 and calculate_accuracy_every_iteration == False):
                    print('Iteration %d/%d - loss: %.4f' %(batch_index, training_length, curr_loss))

            if(calculate_accuracy_every_iteration == False):
                correct = 0
                total = 0
                for images, labels in test_loader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                    correct = correct.data.item()
                accuracy = correct/total
            else:
                accuracy = np.mean(np.array(accuracies))

            print('Epoch : %d/%d - avg_loss: %.4f - avg_acc : %.4f' %(epoch+1, epochs, np.mean(np.array(losses)), accuracy))
        
        dummy_input = torch.randn(1, num_channels, self.__image_height, self.__image_width)
        torch.onnx.export(model, dummy_input, output_model_path)

    def __train_cntk(self, path_to_folder: str, model_definition, epochs: int, output_model_path: str, classes, minibatch_size: int):
        import cntk
        from cntk.learners import learning_parameter_schedule
        from cntk.ops import input_variable
        from cntk.io import MinibatchSource, ImageDeserializer, StreamDefs, StreamDef, MinibatchData, UserDeserializer
        import cntk.io.transforms as xforms
        from cntk.layers import default_options, Dense, Sequential, Activation, Embedding, Convolution2D, MaxPooling, Stabilizer, Convolution, Dropout, BatchNormalization
        from cntk.ops.functions import CloneMethod
        from cntk.logging import ProgressPrinter
        from cntk.losses import cross_entropy_with_softmax
        from cntk import classification_error, softmax, relu, ModelFormat, element_times, momentum_schedule, momentum_sgd
        import pandas as pd

        path_to_folder = path_to_folder.rstrip('/')

        map_file_train = path_to_folder+"/train_map.txt"
        map_file_test = path_to_folder+"/test_map.txt"
        classes_set = set()
        num_train = 0
        num_test = 0
        num_channels = 3

        class TrackDataset(UserDeserializer):
            def __init__(self, map_file, streams, chunksize = 100):
                super(TrackDataset, self).__init__()
                self._batch_size = chunksize
                self.dataframes = pd.read_csv(map_file, sep='\t', dtype=str, header=None, names=["features", "labels"])
                self._streams = [cntk.io.StreamInformation(s['name'], i, 'dense', np.float32, s['shape']) for i, s in enumerate(streams)]

                self._num_chunks = int(math.ceil(len(self.dataframes)/chunksize))

            def _scale_image(self, image, width=224, height=168):
                try:
                    return image.resize((width,height), Image.LINEAR)
                except:
                    raise Exception('scale_image error')  

            def stream_infos(self):
                return self._streams

            def num_chunks(self):
                return self._num_chunks

            def get_chunk(self, chunk_id):
                images = []
                labels = []
                maximum = (chunk_id+1)*self._batch_size
                if(maximum > len(self.dataframes)):
                    maximum = len(self.dataframes)
                for i in range(chunk_id*self._batch_size, maximum):
                    img_name = self.dataframes.iloc[i, 0]
                    image = Image.open(img_name)
                    cl = self.dataframes.iloc[i, 1:].values[0]
                    image = self._scale_image(image)
                    image = np.moveaxis((np.array(image).astype('float32')), -1, 0)
                    image -= np.mean(image, keepdims=True)
                    image /= (np.std(image, keepdims=True) + 1e-6)
                    images.append(image)
                    yv = np.zeros(num_classes)
                    yv[classes.index(cl)] = 1
                    labels.append(yv)

                result = {}
                features = np.array(images)
                lab = np.array(labels).astype('float32')
                result[self._streams[0].m_name] = features
                result[self._streams[1].m_name] = lab
                return result

        try:
            with open(map_file_train) as f:
                csv_reader = csv.reader(f, delimiter='\t')
                for row in csv_reader:
                    cmd = row[1]
                    classes_set.add(cmd)
                    num_train = num_train + 1
        except Exception as e:
            raise Exception("No train_map.txt file found in path "+path_to_folder+". Did you create a dataset using create_balanced_dataset()?")

        num_classes = len(classes)

        with open(map_file_test) as f:
            for num_test, l in enumerate(f):
                pass

        # transforms = [
        #     xforms.scale(width=self.__image_width, height=self.__image_height, channels=num_channels, interpolations='linear'),
        #     xforms.mean(mean_file)
        # ]

        dataset_train = TrackDataset(map_file=map_file_train, streams=[dict(name='features', shape=(num_channels,self.__image_height,self.__image_width)), dict(name='labels', shape=(num_classes,))])
        reader_train = MinibatchSource([dataset_train], randomize=True)

        # a = dataset_train.num_chunks()

        dataset_test = TrackDataset(map_file=map_file_test, streams=[dict(name='features', shape=(num_channels,self.__image_height,self.__image_width)), dict(name='labels', shape=(num_classes,))])
        reader_test = MinibatchSource([dataset_test], randomize=True)

        # ImageDeserializer loads images in the BGR format, not RGB
        # reader_train = MinibatchSource(ImageDeserializer(map_file_train, StreamDefs(
        #     features = StreamDef(field='image', transforms=transforms),
        #     labels   = StreamDef(field='label', shape=num_classes)
        # )))

        # reader_test = MinibatchSource(ImageDeserializer(map_file_test, StreamDefs(
        #     features = StreamDef(field='image', transforms=transforms),
        #     labels   = StreamDef(field='label', shape=num_classes)
        # )))

        # mb = reader_train.next_minibatch(10)

        input_var = input_variable((num_channels, self.__image_height, self.__image_width))
        label_var = input_variable((num_classes))

        model = model_definition(input_var)

        ce = cross_entropy_with_softmax(model, label_var)
        pe = classification_error(model, label_var)

        epoch_size = num_train

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

        print("Training started")
        batch_index = 0
        plot_data = {'batchindex':[], 'loss':[], 'error':[]}
        for epoch in range(epochs):
            sample_count = 0
            while sample_count < epoch_size:
                data: MinibatchSource = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count), input_map=input_map)

                trainer.train_minibatch(data)
                sample_count += data[label_var].num_samples

                batch_index += 1
                plot_data['batchindex'].append(batch_index)
                plot_data['loss'].append(trainer.previous_minibatch_loss_average)
                plot_data['error'].append(trainer.previous_minibatch_evaluation_average)

            trainer.summarize_training_progress()

        metric_numer = 0
        metric_denom = 0
        sample_count = 0
        minibatch_index = 0
        epoch_size = num_test

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


    def __plot_test(self, confusion_matrix):
        #plt.figure(1)
        #plt.subplot(211)
        #plt.plot(plot_data["batchindex"], plot_data["avg_loss"], 'b--')
        #plt.xlabel('Minibatch number')
        #plt.ylabel('Loss')
        #plt.title('Minibatch run vs. Training loss ')

        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        
        df = confusion_matrix.stats()["class"]

        # 10 = recall, 12 = precision, 17 = accuracy, 18 = F1-score
        ndf = df.ix[[10,12,17,18]]

        ndf['Scores'] = ["Recall", "Precision", "Accuracy", "F1-Score"]

        ax.table(cellText=ndf.values, colLabels=ndf.columns, loc='center')
        fig.tight_layout()
        confusion_matrix.plot()
        plt.show()

    def test(self, path_to_model: str, path_to_test_map: str):

        from pandas_ml import ConfusionMatrix

        images = []
        ground_truth = []
        predictions = []
        try:
            with open(path_to_test_map) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='\t')
                for row in csv_reader:
                    images.append(row[0])
                    ground_truth.append(row[1])
        except:
            raise Exception("Could not parse testmap. Did you provide a path to a valid test_map.txt file?")

        sess = rt.InferenceSession(path_to_model)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name

        for i, image in enumerate(images):
            try:
                img = Image.open(image)
            except Exception as e:
                print("Cant read "+image)
            try:
                processed_image = self.__scale_image(img)
            except Exception as e:
                print("Err while reading " + image + ": " + str(e))

            X = np.array([np.moveaxis((np.array(processed_image).astype('float32')), -1, 0)])
            X -= np.mean(X, keepdims=True)
            X /= (np.std(X, keepdims=True) + 1e-6)

            pred = sess.run([label_name], {input_name: X.astype(np.float32)})[0]
            index = int(np.argmax(pred))
            predictions.append(self.__commands[index])

        confusion_matrix = ConfusionMatrix(ground_truth, predictions)
        #print("Confusion matrix:\n%s" % confusion_matrix)

        self.__plot_test(confusion_matrix)
