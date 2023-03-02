import sys
import pandas as pd
import numpy as np
from scipy.signal import medfilt
from scipy import interpolate
from scipy.signal import resample
from sklearn.metrics import mean_squared_error
from math import sqrt

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    epsilon = sys.float_info.epsilon
    return vector / (np.linalg.norm(vector) + epsilon)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2': """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    #c = np.cross(v2_u,v1_u)
    #if(c>0): 
    #    angle+= 180 
    return angle/np.pi * 180

def applyMedianFilter(signal,N=5):
    filtered_signal = medfilt(signal,N)
    return filtered_signal

def fill_nan(A):
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
    B = np.where(np.isfinite(A),A,f(inds))
    return B
 
# to remove first samples of the signal
def cutBeginning(signal,N=5):
    return signal[N:]

# to filter video signal
def applyMovingAverageFilter(signal,N=5):
    filtered_signal = np.convolve(signal.flatten(), np.ones((N,))/N, mode='valid')
    return filtered_signal

# To downsample imu signal
def downsampleSignal(signal,oldHz,newHz=30):
    new_hz = 30
    secs = len(signal) / oldHz
    samples = int(secs*new_hz)
    subsampled_signal = resample(signal,samples)
    return subsampled_signal

def calcRMSE(signalA, signalB):
    return sqrt(mean_squared_error(signalA,signalB))

def rescaleSignal(input,samples,newamplitude=None):
    signal = input[:samples]
    max = np.max(signal)
    min = np.min(signal)
    if newamplitude:
        output = newamplitude*(signal-min)/(max-min)
    else:
        output = max*(signal-min)/(max-min)
    return output, max

def centerSignalInMean(input,samples):
    signal = input[:samples]
    mean = np.mean(signal)
    output = (signal-mean)
    return output