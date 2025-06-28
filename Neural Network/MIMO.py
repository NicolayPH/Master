import tensorflow as tf
import numpy as np
import statistics
import sklearn as sk
from sklearn import metrics
import matplotlib.pyplot as plt
import time
import json
import csv

from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import Dense, Activation, Dropout, InputLayer, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from tensorflow.keras.layers import LeakyReLU, PReLU
from tensorflow.keras.optimizers import Adam

from bayes_opt import BayesianOptimization

#Creating a function that extracts the data

def obtainDataFromFile(component):
    matrices = []
    current_matrix = []

    filename = "PreparedTraining" + component + ".csv"

    with open(filename, 'r') as f:
        for line in f:
            line_stripped = line.strip()
            if line_stripped == '':
                matrices.append(current_matrix)
                current_matrix = []
            else:
                current_matrix.append([float(element) for element in line_stripped.split(',')])
        matrices.append(current_matrix)


    x_input = np.array(matrices[0])
    validation_input = np.array(matrices[1])


    label = np.array(matrices[2])
    validation_label = np.array(matrices[3])

    x_input = x_input.T
    label = label.T


    label_max, label_min = label.max(), label.min()
    print("Label_max:", label_max)
    scaled_label = (label-label_min)/(label_max-label_min)



    validation_input = validation_input.T
    validation_label = validation_label.T


    validation_label_max, validation_label_min = label_max, label_min
    validated_labeled_output = (validation_label-validation_label_min)/(validation_label_max-validation_label_min)



    split_index_input = len(validation_input) // 2
    dev_input = validation_input[:split_index_input]
    test_input = validation_input[split_index_input:]

    split_index_label = len(validated_labeled_output) // 2
    dev_label = validated_labeled_output[:split_index_label]
    test_label = validated_labeled_output[split_index_label:]
    return x_input, scaled_label, dev_input, test_input, dev_label, test_label


component = "CarbonDioxide"
x_input, scaled_label, dev_input, test_input, dev_label, test_label = obtainDataFromFile(component)


def generate_model(dropout, neuronPct, neuronShrink):
    #We begin with a percentage of 5000 neurons for our first layer
    #ref: https://arxiv.org/pdf/2009.05673 
    neuronCount = int(neuronPct * 1000)


    #Construct neural network
    model = tf.keras.models.Sequential()
    layer = 0

    while neuronCount > 25 and layer < 7:
        if layer == 0:
            model.add(Dense(neuronCount, input_shape = (x_input.shape[1], ), activation = 'relu'))
        
        else:
            model.add(Dense(neuronCount, activation = 'relu'))
        layer += 1

        #Add Batch normalization after each layer
        #model.add(BatchNormalization())
        #Add dropout after each hidden layer
        model.add(Dropout(dropout))

        #Decrease the remaining neuron count after each layer
        neuronCount = int(neuronCount * neuronShrink)

    model.add(Dense(scaled_label.shape[1], activation = 'linear')) #output
    return model


SPLITS = 2
EPOCHS = 3000
PATIENCE = 10


def evaluate_network(dropout, learning_rate, neuronPct, neuronShrink):
    #Track progress
    mean_benchmark = []
    epochs_needed = []
    num = 0

    model = generate_model(dropout, neuronPct, neuronShrink)
    model.compile(loss = 'mean_squared_error', optimizer = Adam(learning_rate = learning_rate))
    monitor = EarlyStopping(monitor = 'val_loss', min_delta = 1e-3, patience = PATIENCE, verbose = 0, mode = 'auto', restore_best_weights = True)

    #Train on the bootstrap sample
    model.fit(x_input, scaled_label, validation_data = (dev_input, dev_label), callbacks = [monitor], verbose = 0, epochs = EPOCHS, batch_size = 128)
    epochs = monitor.stopped_epoch
    epochs_needed.append(epochs)

    prediction = model.predict(dev_input)
    score = metrics.mean_squared_error(prediction, dev_label)
    mean_benchmark.append(score)

    #Clear session
    tf.keras.backend.clear_session()

    m1 = statistics.mean(mean_benchmark)
    m2 = statistics.mean(epochs_needed)
    mdev = statistics.pstdev(mean_benchmark)

    return -m1


#Next step is to optimize this process
pbounds = {'dropout': (0.0, 0.499),
           'learning_rate': (0.0, 0.1),
           'neuronPct': (0.01, 1),
           'neuronShrink': (0.01, 1)
           }

optimizer = BayesianOptimization(
    f = evaluate_network,
    pbounds=pbounds,
    verbose = 2,
    random_state = 1,
)

def print_optimized_variables(component, init_points = 10, n_iter = 80):
    start_time = time.time()
    optimizer.maximize(init_points, n_iter)
    time_took = time.time() - start_time
    print("Time took:", time_took)
    print(optimizer.max)
    filename = "optimizer_max_" + component + ".json"

    with open(filename, 'w') as file:
        json.dump(optimizer.max, file, indent = 4)

#print_optimized_variables("Hydrogen")


def create_optimized_model(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    
    parameters = data.get("params")
    dropout = parameters.get("dropout")
    neuronPct = parameters.get("neuronPct")
    neuronShrink = parameters.get("neuronShrink")
    alpha = parameters.get("learning_rate")

    opt_model = generate_model(dropout = dropout, neuronPct = neuronPct, neuronShrink = neuronShrink)
    opt_model.compile(loss = 'mean_squared_error', optimizer = Adam(learning_rate = alpha))

    monitor = EarlyStopping(monitor = 'val_loss', min_delta = 1e-3, patience = PATIENCE, verbose = 0, mode = 'auto', restore_best_weights = True)

    #Train on the bootstrap sample
    opt_model.fit(x_input, scaled_label, validation_data = (dev_input, dev_label), callbacks = [monitor], verbose = 0, epochs = EPOCHS, shuffle = True, batch_size = 128)
    return opt_model


#opt_model = create_optimized_model("optimizer_max_Hydrogen.json")

#Optimal Bayesian parameters derived from the optimalization
#The structure is [Dropout, Neuron Percentage, Neuron Shrinkage, Learning Rate]
opt_params = {"Hydrogen": [[0.07297661940550304, 0.8984631275262264, 0.29169206875647286, 0.010163449147197457],
                           [0.016636186159798092, 0.1149877049819497, 0.6027536506425499, 0.0023145462085277456],
                           [0.0694469621218864, 0.1936504719634992,	0.35251410992317794, 0.015378057562614811]],
              "CarbonDioxide": [[0.369348991507222, 0.16879766605746643, 0.74371114040222, 0.09838879439821609],
                                 [0.17947256838745337, 0.7275879010741872, 0.5257213908147264, 0.05069429027147701],
                                 [0.13755646002041275, 0.5724888941223814, 0.1284396537855462, 0.03929355991914982]],
              "Nitrogen": [[0.3218303736553769, 0.682644371293227, 0.2642750657580455, 0.012522076894588363],
                           [0.00477788338732836, 0.4222469957852342, 0.030467800804570976, 0.08206812375152706],
                           [0.499, 0.19903107317958418,	0.474482349728461, 0.1]]
}

#Want to create plots, so a function is defined
def plot_subplots(ax, true_values, pred_values, legend):
    ax.scatter(true_values.flatten(), pred_values.flatten(), s=2, alpha=0.5)
    ax.text(0.05, 0.95, f"Initalized neurons = {legend}", transform = ax.transAxes, fontsize = 10, verticalalignment = 'top', bbox = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
    ax.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'k--', lw=1)
    ax.set_xlabel(r"Normalized True Values")
    ax.grid(True)

#Plot histogram of data 
def plot_histogram(component):
    #Retrieving data from file
    x_input, scaled_label, dev_input, test_input, dev_label, test_label = obtainDataFromFile(component)

    component_flatten = np.append(scaled_label.flatten(), dev_label.flatten())
    component_flatten = np.append(component_flatten, test_label.flatten())

    #Carbon Dioxide is called "CarbonDioxide", so want to change that in the labeling
    if component == "CarbonDioxide":
        component = "Carbon Dioxide"

    plt.figure()
    plt.hist(component_flatten, bins = 50, color = 'skyblue', edgecolor = 'black')
    plt.title(component + " Density Distribution")
    plt.xlabel('Normalized Density Value')
    plt.ylabel('Frequency')
    plt.grid()
    plt.tight_layout()
    plt.savefig(component + " Histogram.pdf", bbox_inches = "tight")
    plt.show()

#Want to plot with the same style as in LaTeX
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 12,
})  

#Creating functions that write to and read from files, respectively
def writeValuesToFile(filename, pred_values):
    with open(filename, "w", newline = '') as f:
        writeToCSV = csv.writer(f)
        for el in pred_values:
            writeToCSV.writerow(el)

def getValuesFromFile(filename):
    arr = np.loadtxt(filename, delimiter = ',')
    return arr

#Now, we can start saving the data
componentList = list(opt_params.keys())
print(componentList)


#Index, 250 neurons : 0, 1000 neurons : 1 and 5000 neurons :  2
#Making a dictionary to simplify the labelings later
indexes = {0: "250", 
           1: "1000",
           2: "5000"
}



indexList = list(indexes.keys())

def plotAllFigures():
    for component in componentList:
        #The data is different for different component
        x_input, scaled_label, dev_input, test_input, dev_label, test_label = obtainDataFromFile(component)
        for idx in indexList:
            opt_model = generate_model(dropout = opt_params[component][idx][0], neuronPct = opt_params[component][idx][1], neuronShrink = opt_params[component][idx][2])
            opt_model.compile(loss = 'mean_squared_error', optimizer = Adam(learning_rate = opt_params[component][idx][3]))
            monitor = EarlyStopping(monitor = 'val_loss', min_delta = 1e-3, patience = PATIENCE, verbose = 0, mode = 'auto', restore_best_weights = True)

            #Train on the bootstrap sample
            opt_model.fit(x_input, scaled_label, validation_data = (dev_input, dev_label), callbacks = [monitor], verbose = 0, epochs = EPOCHS, shuffle = True, batch_size = 128)
            #print(opt_model(test_input))


            predicted_values = opt_model(test_input).numpy()



            writeValuesToFile(component + "_MIMO_" + indexes[idx] + ".csv", predicted_values)


        model_250 = getValuesFromFile(component + "_MIMO_250.csv")
        model_1000 = getValuesFromFile(component + "_MIMO_1000.csv")
        model_5000 = getValuesFromFile(component + "_MIMO_5000.csv")


        fig, axs = plt.subplots(3, 1, figsize = (6, 8), sharex = True) #figsize=(5, 6), sharex=True, dpi=300)


        # Subplots
        plot_subplots(axs[0], test_label, model_250, "250")
        plot_subplots(axs[1], test_label, model_1000, "1000")
        plot_subplots(axs[2], test_label, model_5000, "5000")

        axs[1].set_ylabel(r"Normalized Predictions")
        axs[2].set_xlabel(r"Normalized True Values")

        for ax in axs[:2]:
            ax.label_outer()

        #plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.savefig(component + "_MIMO.pdf", bbox_inches='tight', dpi = 300)  # For LaTeX import
        plt.show()

#plotAllFigures()


#Want to create the histograms
for component in componentList:
    plot_histogram(component)

