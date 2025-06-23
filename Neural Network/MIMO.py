import tensorflow as tf
import numpy as np
import statistics
import sklearn as sk
from sklearn import metrics
import matplotlib.pyplot as plt
import time
import json

from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import Dense, Activation, Dropout, InputLayer, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from tensorflow.keras.layers import LeakyReLU, PReLU
from tensorflow.keras.optimizers import Adam

from bayes_opt import BayesianOptimization

matrices = []
current_matrix = []

with open('PreparedTrainingHydrogen.csv', 'r') as f:
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


print(x_input.shape)
print(label.shape)
print(len(scaled_label.flatten()) + len(validated_labeled_output.flatten()))


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

    model.add(Dense(label.shape[1], activation = 'linear')) #output
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
    filename = "optimizer_max_" + component + ".json"

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
    opt_model.fit(x_input, scaled_label, validation_data = (dev_input, dev_label), callbacks = [monitor], verbose = 0, epochs = EPOCHS, shuffle = True, batch_size = 128)
    return opt_model


opt_model = create_optimized_model("optimizer_max_Hydrogen.json")
'''
'''
opt_model = generate_model(dropout = 0.13755646002041275, neuronPct = 0.5724888941223814, neuronShrink = 0.1284396537855462)
opt_model.compile(loss = 'mean_squared_error', optimizer = Adam(learning_rate = 0.03929355991914982))
monitor = EarlyStopping(monitor = 'val_loss', min_delta = 1e-3, patience = PATIENCE, verbose = 0, mode = 'auto', restore_best_weights = True)

#Train on the bootstrap sample
opt_model.fit(x_input, scaled_label, validation_data = (dev_input, dev_label), callbacks = [monitor], verbose = 0, epochs = EPOCHS, shuffle = True, batch_size = 128)
#print(opt_model(test_input))

#print(err)
plt.scatter(test_label, opt_model(test_input))
plt.xlabel('Normalized True Values')
plt.ylabel('Normalized Predictions')
plt.title("Carbon Dioxide")
plt.plot([0,1], [0,1], 'k--')
plt.savefig("CarbonDioxide_MIMO_5000.pdf")
plt.show()
'''


component_flatten = np.append(scaled_label.flatten(), validated_labeled_output.flatten())
plt.figure()
plt.hist(component_flatten, bins = 50, color = 'skyblue', edgecolor = 'black')
plt.title('Hydrogen Density Distribution')
plt.xlabel('Normalized Density Value')
plt.ylabel('Frequency')
plt.grid()
plt.savefig("Hydrogen Histogram.pdf")
plt.show()