from sklearn.model_selection import train_test_split
from cntk.learners import learning_parameter_schedule, adam, UnitType, learning_rate_schedule
from cntk.ops import input_variable
from cntk.layers import Dense, Sequential, Activation, Embedding, Convolution2D, MaxPooling, Stabilizer, Convolution, Dropout, BatchNormalization
from cntk.ops import sequence, load_model, combine, input_variable
from cntk.ops.functions import CloneMethod
from PIL.ImageOps import equalize
from cntk.logging.graph import find_by_name, get_node_outputs
from PIL import Image
from cntk.logging import ProgressPrinter
from cntk.losses import cosine_distance, cross_entropy_with_softmax, squared_error, binary_cross_entropy
from cntk.initializer import glorot_uniform, glorot_normal
from cntk.train import Trainer
from cntk import classification_error, sgd, softmax, tanh, relu, ModelFormat, placeholder
import numpy as np
import matplotlib.pyplot as plt
import csv
import onnxruntime as rt
import tensorflow as tf
from tensorflow.python.platform import gfile

def pad_image(image):
    target_size = max(image.size)
    result = Image.new('RGB', (target_size, target_size), "black")
    try:
        result.paste(image, (int((target_size - image.size[0]) / 2), int((target_size - image.size[1]) / 2)))
    except:
        print("Error on image " + image)
        raise Exception('pad_image error')
    return result

def normalize(arr):
    arr = arr.astype('float')
    for i in range(3):
        mean = arr[...,i].mean()
        std = arr[...,i].std()
        arr[...,i] = (arr[...,i] - mean)/std
    return arr

def scale_image(image):
    try:
        return image.resize((224,224))
    except:
        raise Exception('pad_image error')


def plot(image_array):
    plt.imshow(image_array)
    plt.show()

def create_balanced_dataset(num_classes):
    X_values = []
    y_values = []

    command_counter_start = {}
    command_counter = {}

    with open("src/ml/data/merged/training.csv") as training_file:
        csv_reader = csv.reader(training_file, delimiter=';')

        for row in csv_reader:
            command = row[1]
            if(command_counter_start.get(command, 0) == 0):
                command_counter_start[command] = 1
            else:
                command_counter_start[command] = command_counter_start[command] + 1

        training_file.seek(0)

        # Get the class with the most samples
        max_class = max(command_counter_start, key = lambda x: command_counter_start.get(x))
        maximum = command_counter_start[max_class]

        for row in csv_reader:
            image = row[0]
            command = row[1]

            if(command_counter.get(command, 0) == 0):
                command_counter[command] = 1
            else:
                command_counter[command] = command_counter[command] + 1

            try:
                img = Image.open("src/ml/data/merged/"+image)
            except Exception as e:
                print("Cant read "+image)
                continue
            try:
                processed_image = normalize(np.array(scale_image(pad_image(img))))
            except:
                continue
            X_values.append(np.moveaxis(np.array(processed_image), -1, 0))
            # X_values.append(np.array(processed_image))
            # plot(processed_image)

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

        # Randomly duplicate classes from minorities to balance classes
        finished = False
        while not finished:
            training_file.seek(0)
            for row in csv_reader:
                image = row[0]
                command = row[1]

                if(command_counter[command] < maximum):
                    command_counter[command] = command_counter[command] + 1
                    try:
                        img = Image.open("src/ml/data/merged/"+image)
                    except Exception as e:
                        print("Cant read "+image)
                        continue
                    try:
                        processed_image = normalize(np.array(scale_image(pad_image(img))))
                    except:
                        continue
                    X_values.append(np.moveaxis(np.array(processed_image), -1, 0))
                    # X_values.append(np.array(processed_image))
                    # plot(processed_image)

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

            finished = True
            for key, value in command_counter.items():
                if(value < maximum):
                    finished = False
                    break

    return np.array(X_values).astype(np.float32), np.array(y_values).astype(np.float32)



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

    start_image = 1155
    end_image = 1230
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

    sess = rt.InferenceSession("car_cntk.model")
    print(sess.get_inputs()[0].shape)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    for i in range(start_image, end_image+1):

        image = "snapshot_"+str(i)+".png"

        try:
            img = Image.open("src/ml/data/merged/"+image)
        except Exception as e:
            print("Cant read "+image)
        try:
            processed_image = equalize(scale_image(img))
        except:
            print("Err while reading " + image)

        X = np.array([np.moveaxis(np.array(processed_image), -1, 0)])/255.0
        #print(X.shape)

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

        X = np.array([np.moveaxis(np.array(processed_image), -1, 0)])/255.0
        #print(X.shape)
        model = load_model("car_cntk_3.model", format=ModelFormat.ONNX)
        result = model.eval({model.arguments[0]: X})

        print("Label    :", y[num])
        print("Predicted:", result)
        num = num + 1


def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"
    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose:
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error*100))
    return mb, training_loss, eval_error


def create_model_pretrained(num_classes, input_features, freeze=False):
    # Load the pretrained classification net and find nodes
    base_model = load_model("C:/Users/chvorhem/Desktop/ML-Garage/ResNet18_ImageNet_CNTK.model")
    feature_node = find_by_name(base_model, "features")
    last_node = find_by_name(base_model, "z.x")

    # Clone the desired layers with fixed weights
    cloned_layers = combine([last_node.owner]).clone(
        CloneMethod.freeze if freeze else CloneMethod.clone,
        {feature_node: placeholder(name='features')})

    # Add new dense layer for class prediction
    #feat_norm  = input_features - Constant(114)
    #cloned_out = cloned_layers(feat_norm)

    cloned_out = cloned_layers(input_features)

    z = Dense(num_classes, activation=softmax, name="prediction") (cloned_out)

    return z


def train():

    num_classes = 3
    X_values, y_values = create_balanced_dataset(num_classes)

    #X_values = np.moveaxis(X_values, 0, -1)
    #y_values = np.moveaxis(y_values, -1, 0)

    np.random.seed(7)

    a = X_values[44]
    #ima = Image.fromarray(a.astype('uint8'))

    X_train, X_test, y_train, y_test = train_test_split(X_values, y_values, test_size=0.2)

    #X_train = X_train.reshape(len(X_train), -1)
    #X_test = X_test.reshape(len(X_test), -1)

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

    print("-------")
    print(X_train.shape)
    print(y_train.shape)

    # plt.imshow(X_train[4])
    # plt.show()

    feature = input_variable(shape=(3, 224, 224))
    label = input_variable(num_classes)

    #model = create_model_pretrained(3, feature)
    model = create_model(feature)

    #feature_nn = input_variable(shape=(57600))
    #model = create_model_nn(feature_nn)

    learning_rate = 0.1
    lr_schedule = learning_rate_schedule(0.001, UnitType.sample)
    learner = sgd(model.parameters, lr_schedule)
    #learner = adam(model.parameters, learning_rate, 0.05)

    loss = cross_entropy_with_softmax(model, label)
    error = classification_error(model, label)
    progress_printer = ProgressPrinter(10)
    trainer = Trainer(model, (loss, error), learner, progress_printer)

    # scenario 1 ------------------------------------------------------------------

    trainingloss_summary = []
    testerror = []
    max_epochs = 2
    minibatch_size = 70

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

            end = end + minibatch_size

        training_loss = trainer.previous_minibatch_loss_average
        trainingloss_summary.append(training_loss)

        # Returns classification error
        t = trainer.test_minibatch({feature : X_test, label : y_test})
        testerror.append(t)

        print("epoch #{}: training_loss={}, test_error={}".format(epoch, trainingloss_summary[-1], testerror[-1]))


    epoch_count = range(0, len(trainingloss_summary))
    plt.plot(trainingloss_summary, label='training loss')
    plt.plot(epoch_count, trainingloss_summary, 'r--')
    plt.plot(epoch_count, testerror, 'b-')
    plt.legend(['Training Loss', 'Test accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    model.save("car_cntk.model", format=ModelFormat.ONNX)

    # scenario 2 -----------------------------------------------------------------

    # minibatch_size = 80
    # num_minibatches = len(X_train) // minibatch_size
    # num_passes = 30
    # training_progress_output_freq = 1
    # plotdata = {"batchsize":[], "loss":[], "error":[]}

    # tf = np.array_split(X_train, num_minibatches)
    # tl = np.array_split(y_train, num_minibatches)

    # for i in range(num_minibatches*num_passes): # multiply by the
    #     features = np.ascontiguousarray(tf[i%num_minibatches])
    #     labels = np.ascontiguousarray(tl[i%num_minibatches])

    #     trainer.train_minibatch({feature : features, label : labels})
    #     batchsize, loss, error = print_training_progress(trainer, i, training_progress_output_freq, verbose=1)
    #     if not (loss == "NA" or error =="NA"):
    #         plotdata["batchsize"].append(batchsize)
    #         plotdata["loss"].append(loss)
    #         plotdata["error"].append(error)

    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(plotdata["batchsize"], plotdata["loss"], 'b--')
    # plt.xlabel('Minibatch number')
    # plt.ylabel('Loss')
    # plt.title('Minibatch run vs. Training loss ')

    # plt.subplot(212)
    # plt.plot(plotdata["batchsize"], plotdata["error"], 'r--')
    # plt.xlabel('Minibatch number')
    # plt.ylabel('Label Prediction Error')
    # plt.title('Minibatch run vs. Label Prediction Error ')
    # plt.show()

    # model.save("car_cntk.model", format=ModelFormat.ONNX)

train()
#test_onnx()