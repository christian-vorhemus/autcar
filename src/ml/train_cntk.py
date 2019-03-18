from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from cntk.learners import learning_parameter_schedule, adam, UnitType, learning_rate_schedule
from cntk.ops import input_variable
from cntk.layers import Dense, Sequential, Activation, Embedding, Convolution2D, MaxPooling, Stabilizer, Convolution, Dropout
from cntk.ops import sequence, load_model
from PIL.ImageOps import equalize
from PIL import Image
from cntk.logging import ProgressPrinter
from cntk.losses import cosine_distance, cross_entropy_with_softmax, squared_error, binary_cross_entropy
from cntk.initializer import glorot_uniform, glorot_normal
from cntk.train import Trainer
from cntk import classification_error, sgd, softmax, tanh, relu, ModelFormat
import numpy as np
import matplotlib.pyplot as plt
import csv
import onnxruntime as rt
import tensorflow as tf
from tensorflow.python.platform import gfile

def pad_image(image):
    target_size = max(image.size)
    result = Image.new('RGB', (target_size, target_size), "white")
    try:
        result.paste(image, (int((target_size - image.size[0]) / 2), int((target_size - image.size[1]) / 2)))
    except:
        print("Error on image " + image)
        raise Exception('pad_image error')
    return result

def scale_image(image):
    try:
        return image.resize((128,128))
    except:
        raise Exception('pad_image error')

def create_dataset(num_classes):
    X_values = []
    y_values = []

    command_counter = {}

    with open("src/ml/data/merged/training.csv") as training_file:
        csv_reader = csv.reader(training_file, delimiter=';')

        for row in csv_reader:
            command = row[1]
            if(command_counter.get(command, 0) == 0):
                command_counter[command] = 1
            else:
                command_counter[command] = command_counter[command] + 1

        training_file.seek(0)

        # Get the class with the fewest samples
        min_class = min(command_counter, key = lambda x: command_counter.get(x))
        minimum = command_counter[min_class]
        command_counter = {}

        for row in csv_reader:
            image = row[0]
            command = row[1]

            if(command_counter.get(command, 0) == 0):
                command_counter[command] = 1
            else:
                command_counter[command] = command_counter[command] + 1

            if(command_counter[command] > minimum):
                continue

            try:
                img = Image.open("src/ml/data/merged/"+image)
            except Exception as e:
                print("Cant read "+image)
                continue
            try:
                processed_image = equalize(scale_image(pad_image(img)))
            except:
                continue
            X_values.append(np.moveaxis(np.array(processed_image), -1, 0))
            # X_values.append(np.array(processed_image))

            #y_values.append(command)
            # Add 3 because we have 3 classes (forward, left, right)
            yv = np.zeros(num_classes)
            if(command == "move"):
                yv[0] = 1
            if(command == "left"):
                yv[1] = 1
            if(command == "right"):
                yv[2] = 1
            y_values.append(yv)

    return np.array(X_values)/255.0, np.array(y_values)



def test_tensorflow():
    # Convert ONNX to tensorflow with:
    # onnx-tf convert -t tf -i car_cntk.model -o car_tensorflow.model

    start_image = 3321
    end_image = 3344
    ground_truth = []

    with open("src/ml/data/merged/training.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        capture = False
        for row in csv_reader:
            if(row[0] == "snapshot_"+str(start_image)+".png"):
                capture = True
            if(capture):
                ground_truth.append(row[1])
            if(row[0] == "snapshot_"+str(end_image)+".png"):
                break

    with tf.Session() as sess:
        print("load graph")
        with gfile.FastGFile('car_tensorflow.model', 'rb') as f:
            graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        graph_nodes=[n for n in graph_def.node]
        names = []
        for t in graph_nodes:
            names.append(t.name)
        #print(names)

        softmax_tensor = sess.graph.get_tensor_by_name('Softmax612:0')

        for i in range(start_image, end_image+1):

            image = "snapshot_"+str(i)+".png"

            try:
                img = Image.open("src/ml/data/merged/"+image)
            except Exception as e:
                print("Cant read "+image)
            try:
                processed_image = equalize(scale_image(pad_image(img)))
            except Exception as e:
                print("Err while reading " + image)

            X = np.array([np.moveaxis(np.array(processed_image), -1, 0)])/255.0

            pred = sess.run(softmax_tensor, {'Input501:0': X})
            index = np.argmax(pred)

            if(index == 0):
                prediction = "move"
            elif(index == 1):
                prediction = "left"
            elif(index == 2):
                prediction = "right"

            print(image + ". Ground truth: " + ground_truth[i-start_image] + ", Prediction: " + prediction)


def test_onnx():

    start_image = 3321
    end_image = 3344
    ground_truth = []

    with open("src/ml/data/merged/training.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        capture = False
        for row in csv_reader:
            if(row[0] == "snapshot_"+str(start_image)+".png"):
                capture = True
            if(capture):
                ground_truth.append(row[1])
            if(row[0] == "snapshot_"+str(end_image)+".png"):
                break


    for i in range(start_image, end_image+1):

        image = "snapshot_"+str(i)+".png"

        try:
            img = Image.open("src/ml/data/merged/"+image)
        except Exception as e:
            print("Cant read "+image)
        try:
            processed_image = equalize(scale_image(pad_image(img)))
        except:
            print("Err while reading " + image)

        X = np.array([np.moveaxis(np.array(processed_image), -1, 0)])/255.0

        sess = rt.InferenceSession("car_cntk.model")
        #print(sess.get_inputs()[0].shape)
        #print(X.shape)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        pred = sess.run([label_name], {input_name: X.astype(np.float32)})[0]
        index = np.argmax(pred)

        if(index == 0):
            prediction = "move"
        elif(index == 1):
            prediction = "left"
        elif(index == 2):
            prediction = "right"

        print(image + ". Ground truth: " + ground_truth[i-start_image] + ", Prediction: " + prediction)



def test():
    # left, right, forward
    images = ["snapshot_34.png", "snapshot_93.png", "snapshot_98.png"]
    y = np.array([[0,1,0],[0,0,1],[1,0,0]], dtype=np.float32)

    num = 0
    for image in images:
        try:
            img = Image.open("src/ml/data/merged/"+image)
        except Exception as e:
            print("Cant read "+image)
        try:
            processed_image = equalize(scale_image(pad_image(img)))
        except:
            print("Err while reading " + image)

        X = np.array(np.moveaxis(np.array(processed_image), -1, 0))/255.0
        model = load_model("car_cntk.model", format=ModelFormat.ONNX)
        result = model.eval({model.arguments[0]: X})

        print("Label    :", y[num])
        print("Predicted:", result)
        num = num + 1



def train():

    num_classes = 3
    X_values, y_values = create_dataset(num_classes)

    #X_values = np.moveaxis(X_values, 0, -1)
    #y_values = np.moveaxis(y_values, -1, 0)

    np.random.seed(7)

    X_train, X_test, y_train, y_test = train_test_split(X_values, y_values, test_size=0.2)

    create_model = Sequential([
        Convolution2D(filter_shape=(5,5), num_filters=32, strides=(1,1), pad=True, name="first_conv", activation=relu),
        MaxPooling(filter_shape=(3,3), strides=(2,2), name="first_max"),
        Convolution2D(filter_shape=(3,3), num_filters=48, strides=(1,1), pad=True, name="second_conv", activation=relu),
        MaxPooling(filter_shape=(3,3), strides=(2,2), name="second_max"),
        Convolution2D(filter_shape=(3,3), num_filters=64, strides=(1,1), pad=True, name="third_conv", activation=relu),
        Dense(100, activation=relu),
        Dropout(0.1),
        Dense(num_classes, activation=softmax)
    ])

    print("-------")
    print(X_train.shape)
    print(y_train.shape)

    # plt.imshow(X_train[4])
    # plt.show()

    feature = input_variable(shape=(3, 128, 128))
    label = input_variable(num_classes)

    model = create_model(feature)

    learning_rate = 0.1
    lr_schedule = learning_rate_schedule(0.001, UnitType.sample)
    learner = sgd(model.parameters, lr_schedule)
    #learner = adam(model.parameters, learning_rate, 0.05)

    loss = cross_entropy_with_softmax(model, label)
    error = classification_error(model, label)
    progress_printer = ProgressPrinter(10)
    trainer = Trainer(model, (loss, error), learner, progress_printer)
    trainingloss_summary = []
    testerror = []

    max_epochs = 14
    minibatch_size = 80

    # Number of epochs in for-loop. 
    for epoch in range(max_epochs):

        end = minibatch_size
        running = True

        while running:

            if(end >= len(X_train)):
                end = len(X_train)-1
                running = False

            X_current = X_train[end-minibatch_size:end]
            y_current = y_train[end-minibatch_size:end]

            trainer.train_minibatch({feature : X_current, label : y_current})
            if(epoch % 5 == 0):
                training_loss = trainer.previous_minibatch_loss_average
                trainingloss_summary.append(training_loss)
                t = trainer.test_minibatch({feature : X_test, label : y_test})
                testerror.append(t)
        
            end = end + minibatch_size

    epoch_count = range(0, len(trainingloss_summary))
    plt.plot(trainingloss_summary, label='training loss')
    plt.plot(epoch_count, trainingloss_summary, 'r--')
    plt.plot(epoch_count, testerror, 'b-')
    plt.legend(['Training Loss', 'Test accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    model.save("car_cntk.model", format=ModelFormat.ONNX)

train()