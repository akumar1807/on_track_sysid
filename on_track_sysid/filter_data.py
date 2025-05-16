import numpy as np
from scipy.signal import butter, filtfilt

def filter_data(training_data, model):
    """
    Filter training data.

    Applies a low-pass Butterworth filter to each column of the training data array.
    """
    b, a = butter(N=3, Wn=0.1, btype='low')
    #print(b,a)
    #print(training_data)
    training_data[:,0] = filtfilt(b, a, training_data[:,0]) # v_x, longitudinal velocity
    training_data[:,1] = filtfilt(b, a, training_data[:,1]) # v_y, lateral velocity
    training_data[:,2] = filtfilt(b, a, training_data[:,2]) # omega, yaw rate
    training_data[:,3] = filtfilt(b, a, training_data[:,3]) # delta, steering angle
    training_data[:,3] = np.roll(training_data[:,3], 5)
    training_data = training_data[5:,:]
    
    
    '''# If the model is not a simulation, adjusts lateral velocity based on the car's rear axle length.
    if model["racecar_version"] != "SIM":
        training_data[:,1] = training_data[:,1] + model["l_r"] * training_data[:,2]'''
    
    return training_data

def negate_data(training_data):
    """
    Negate training data along the y-axis.

    Negates the lateral velocity (v_y), yaw rate (omega) and steering angle (delta) components of the training data
    to generate additional training samples with the opposite direction.
    """
    negate_data = np.zeros((2*training_data.shape[0], training_data.shape[1]))
    negate_data[:,0] = np.append(training_data[:,0], training_data[:,0])
    negate_data[:,1] = np.append(training_data[:,1], -training_data[:,1])
    negate_data[:,2] = np.append(training_data[:,2], -training_data[:,2])
    negate_data[:,3] = np.append(training_data[:,3], -training_data[:,3])
    
    return negate_data

def process_data(training_data, model):
    """
    Process training data.
    Filters and negates the training data to prepare it for training the neural network.
    """
    filtered_data = filter_data(training_data, model)
    negated_data = negate_data(filtered_data)
    
    return negated_data