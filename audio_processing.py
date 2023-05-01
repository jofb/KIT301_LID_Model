import scipy.signal
import miniaudio
import numpy as np
from scipy import linalg, fft as sp_fft

def custom_resample(x, num, t=None, axis=0, window=None, domain='time'):
    x = np.asarray(x)
    Nx = x.shape[axis] 

    # first fft
    X = sp_fft.rfft(x, axis=axis) 
    
    newshape = list(x.shape) 
    newshape[0] = num // 2 + 1

    Y = np.zeros(newshape, X.dtype) 
    
    Nx = x.shape[axis]
    
    N = min(num, Nx)
    
    nyq = N // 2 + 1
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, nyq)
    Y[tuple(sl)] = X[tuple(sl)]

    if N % 2 == 0:
        if num < Nx:
            sl[axis] = slice(N // 2, N // 2 + 1)
            Y[tuple(sl)] *= 2
    
    newy = np.pad(Y, (0, nyq), mode='constant')
    #print(Y)
    y = sp_fft.irfft(Y, len(Y), axis=axis) # len(Y) != 140000; pad Y with zeros until its length 140000
    print(y)
    
    # print(y)
    # print (np.fft.irfft(Y, num, axis=axis))
    # y *= (float(num) / float(Nx))
    
    custom_irfft(newy, num, axis=axis)



def custom_irfft(a, n, axis=-1, norm=None):
    a = np.core.asarray(a)
    # output = raw_fft(a, n, axis, True, False, n)
    raw_fft(a, n, axis, True, False, n)
    # return output

def raw_fft(a, n, axis, is_real, is_forward, inv_norm):
    # axis = normalize_axis_index(axis, a.ndim)
    fct = 1/inv_norm
    
    if a.shape[axis] != n:
        s = list(a.shape)
        index = [slice(None)]*len(s)
        if s[axis] > n:
            index[axis] = slice(0, n)
            a = a[tuple(index)]
        else: # pad
            index[axis] = slice(0, s[axis])
            s[axis] = n
            z = np.zeros(s, a.dtype.char)
            z[tuple(index)] = a
            a = z

    # r = pfi.execute(a, is_real, is_forward, fct)
    # return r


def read_mp3(path, resample_rate=16000):
    if isinstance(path, bytes):
        # If path is a tf.string tensor, it will be in bytes
        path = path.decode("utf-8")
        
    f = miniaudio.mp3_read_file_f32(path)
    
    # Downsample to target rate, 16 kHz is commonly used for speech data
    new_len = round(len(f.samples) * float(resample_rate) / f.sample_rate)
    signal = scipy.signal.resample(f.samples, new_len)
    
    custom_resample(f.samples, new_len)
    

    # Normalize to [-1, 1]
    # signal /= np.abs(signal).max()

    # return signal, resample_rate


# signal, rate = read_mp3('./testinput/et-test (1).mp3')
read_mp3('./testinput/et-test (1).mp3')