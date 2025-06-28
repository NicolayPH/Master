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


#For the MISO approach, each input layer needs to be added the numbers 1-201. 
#Thus, each row must be repeated 201 times

def add_grid_numbers(input):
        append_values = np.arange(1, 202)
        repeated_input = np.repeat(input, repeats = 201, axis = 0)

        #Now, we can make the z column that will be appended
        z_column = np.tile(append_values, reps = input.shape[0]).reshape(-1, 1)
        final_input = np.hstack((repeated_input, z_column))
        return final_input

#Creating a function that extracts the data

def obtainDataFromFile(filename):
    matrices = []
    current_matrix = []

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
    scaled_label = (label-label_min)/(label_max-label_min)


    validation_input = validation_input.T
    validation_label = validation_label.T

    x_input = add_grid_numbers(x_input)
    validation_input = add_grid_numbers(validation_input)


    validation_label_max, validation_label_min = label_max, label_min
    validated_labeled_output = (validation_label-validation_label_min)/(validation_label_max-validation_label_min)


    validated_labeled_output = validated_labeled_output.flatten().reshape(-1, 1)
    scaled_label = scaled_label.flatten().reshape(-1,1)

    split_index_input = len(validation_input) // 2
    dev_input = validation_input[:split_index_input]
    test_input = validation_input[split_index_input:]

    split_index_label = len(validated_labeled_output) // 2
    dev_label = validated_labeled_output[:split_index_label]
    test_label = validated_labeled_output[split_index_label:]
    return x_input, scaled_label, dev_input, test_input, dev_label, test_label


filename = 'PreparedTrainingCarbonDioxide.csv' 
x_input, scaled_label, dev_input, test_input, dev_label, test_label = obtainDataFromFile(filename)
print(scaled_label)

print(x_input.shape)
print(scaled_label.shape)
print(dev_label.shape)


def generate_model(dropout, neuronPct, neuronShrink):
    #We begin with a percentage of 5000 neurons for our first layer
    #ref: https://arxiv.org/pdf/2009.05673 
    neuronCount = int(neuronPct * 5000)


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
        model.add(BatchNormalization())
        #Add dropout after each hidden layer
        model.add(Dropout(dropout))

        #Decrease the remaining neuron count after each layer
        neuronCount = int(neuronCount * neuronShrink)

    model.add(Dense(scaled_label.shape[1], activation = 'linear')) #output
    return model


SPLITS = 2
EPOCHS = 1000
PATIENCE = 10


def evaluate_network(dropout, learning_rate, neuronPct, neuronShrink):
    #boot = ShuffleSplit(n_splits = SPLITS, test_size = 0.1)

    #Track progress
    mean_benchmark = []
    epochs_needed = []
    num = 0

    model = generate_model(dropout, neuronPct, neuronShrink)
    model.compile(loss = 'mean_squared_error', optimizer = Adam(learning_rate = learning_rate))
    monitor = EarlyStopping(monitor = 'val_loss', min_delta = 1e-3, patience = PATIENCE, verbose = 0, mode = 'auto', restore_best_weights = True)

    #Train on the bootstrap sample
    model.fit(x_input, scaled_label, validation_data = (dev_input, dev_label), callbacks = [monitor], verbose = 0, epochs = EPOCHS, shuffle = True, batch_size = 128)
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
'''

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
    filename = "optimizer_max_MISO_" + component + ".json"

    with open(filename, 'w') as file:
        json.dump(optimizer.max, file, indent = 4)

print_optimized_variables("Hydrogen")


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
    opt_model.fit(x_input, scaled_label, validation_data = (dev_input, dev_label), callbacks = [monitor], verbose = 0, epochs = EPOCHS, batch_size = 128)
    return opt_model


opt_model = create_optimized_model("optimizer_max_MISO_Hydrogen.json")
'''


#Optimal Bayesian parameters derived from the optimalization
#The structure is [Dropout, Neuron Percentage, Neuron Shrinkage, Learning Rate]
opt_params = {"Hydrogen": [[0.1848743322126204, 0.4743176979592553, 0.778096724897786, 0.04027761491347523],
                          [0.16837564583758236, 0.5416705506200448,	0.3721733266564638, 0.06566139574322541],
                            [0.19798696411043, 0.4250025692592619,	0.6883673053927919, 0.0538816734003357]],
              "CarbonDioxide": [[0.432914747407662, 0.717145422776755, 0.3341282059666561, 0.0800159045996249],
                               [0.3976476431269369, 0.7964735689412817, 0.7283293722152454, 0.08915556939599203]],
              "Nitrogen": [[0.11200877459831117, 0.5725772716455936, 0.953133835030213, 0.052770534789438155],
                          [0.09284031889436505, 0.42654441107643976, 0.08369921893509626, 0.11970898251567202],
                          [0.3525897295394852, 0.7794703419904637, 0.3468746147832975, 0.05550026809326039]]
}


#Want to create plots, so a function is defined
def plot_subplots(ax, true_values, pred_values, legend):
    ax.scatter(true_values.flatten(), pred_values.flatten(), s=2, alpha=0.5)
    ax.text(0.05, 0.95, f"Initalized neurons = {legend}", transform = ax.transAxes, fontsize = 10, verticalalignment = 'top', bbox = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
    ax.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'k--', lw=1)
    ax.set_xlabel(r"Normalized True Values")
    ax.grid(True)

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
        preparedTraining_filename = "PreparedTraining" + component + ".csv"
        x_input, scaled_label, dev_input, test_input, dev_label, test_label = obtainDataFromFile(preparedTraining_filename)
        if component == "CarbonDioxide":   #Since the data for 5000 isn't available for Carbon Dioxide
            indexList = [0, 1]
        else: 
            indexList = list(indexes.keys())
        for idx in indexList:
            opt_model = generate_model(dropout = opt_params[component][idx][0], neuronPct = opt_params[component][idx][1], neuronShrink = opt_params[component][idx][2])
            opt_model.compile(loss = 'mean_squared_error', optimizer = Adam(learning_rate = opt_params[component][idx][3]))
            monitor = EarlyStopping(monitor = 'val_loss', min_delta = 1e-3, patience = PATIENCE, verbose = 0, mode = 'auto', restore_best_weights = True)

            #Train on the bootstrap sample
            opt_model.fit(x_input, scaled_label, validation_data = (dev_input, dev_label), callbacks = [monitor], verbose = 0, epochs = EPOCHS, shuffle = True, batch_size = 128)
            #print(opt_model(test_input))


            predicted_values = opt_model(test_input).numpy()



            writeValuesToFile(component + "_MISO_" + indexes[idx] + ".csv", predicted_values)


        model_250 = getValuesFromFile(component + "_MISO_250.csv")
        model_1000 = getValuesFromFile(component + "_MISO_1000.csv")

        
        
        if component != "CarbonDioxide":
            model_5000 = getValuesFromFile(component + "_MISO_5000.csv")
            fig, axs = plt.subplots(3, 1, figsize = (6, 8), sharex = True) #figsize=(5, 6), sharex=True, dpi=300)
        else:
            fig, axs = plt.subplots(2, 1, figsize = (6, 8), sharex = True) #figsize=(5, 6), sharex=True, dpi=300)

        # Subplots
        plot_subplots(axs[0], test_label, model_250, "250")
        plot_subplots(axs[1], test_label, model_1000, "1000")
        if component != "CarbonDioxide":
            plot_subplots(axs[2], test_label, model_5000, "5000")
            axs[1].set_ylabel(r"Normalized Predictions")
            axs[2].set_xlabel(r"Normalized True Values")
        else:
            #axs[1].set_ylabel(r"Normalized Predictions")
            fig.text(0.04, 0.5, 'Normalized Predictions', va='center', rotation='vertical', fontsize=12)
            axs[1].set_xlabel(r"Normalized True Values")

        

        
        for ax in axs[:2]:
            ax.label_outer()

        #plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.savefig(component + "_MISO.pdf", bbox_inches='tight', dpi = 300)  # For LaTeX import
        plt.show()

plotAllFigures()


