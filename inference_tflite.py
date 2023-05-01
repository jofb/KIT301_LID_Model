import os
import tensorflow as tf
import numpy as np
import miniaudio
from lidbox.features.audio import spectrograms
from lidbox.features.audio import linear_to_mel
from lidbox.features.audio import framewise_rms_energy_vad_decisions
from lidbox.features import cmvn
import scipy.signal
import time

'''
Benchmark results on 18/04 with a single audio file: Processing : ~ 0.8 sec. Inference: ~0.01 sec.
'''

AUDIO_PATH = "./testinput/et-test (1).mp3"
MODEL_PATH = "./tflite_model.tflite"


''' AUDIO PIPELINE '''
''' The following functions are used to process the data ready for input '''

# Reads in an mp3 file
# adapted from lidbox tutorial
def read_mp3(path, resample_rate=16000):
    if isinstance(path, bytes):
        # If path is a tf.string tensor, it will be in bytes
        path = path.decode("utf-8")
        
    f = miniaudio.mp3_read_file_f32(path)
    
    # Downsample to target rate, 16 kHz is commonly used for speech data
    new_len = round(len(f.samples) * float(resample_rate) / f.sample_rate)
    signal = scipy.signal.resample(f.samples, new_len)
    

    # Normalize to [-1, 1]
    signal /= np.abs(signal).max()

    return signal, resample_rate

# converts signals to a logmel spectrogram, used as final input for model
# adapted from lidbox tutorial
def logmelspectrograms(signals, rate):
    powspecs = spectrograms(signals, rate)
    melspecs = linear_to_mel(powspecs, rate, num_mel_bins=40)
    return tf.math.log(melspecs + 1e-6)

# voice activity detection
# adapted from lidbox tutorial
def remove_silence(signal, rate):
    window_ms = tf.constant(10, tf.int32)
    window_frames = (window_ms * rate) // 1000
    
    # Get binary VAD decisions for each 10 ms window
    vad_1 = framewise_rms_energy_vad_decisions(
        signal=signal,
        sample_rate=rate,
        frame_step_ms=window_ms,
        # Do not return VAD = 0 decisions for sequences shorter than 300 ms
        min_non_speech_ms=300,
        strength=0.1)
    
    # Partition the signal into 10 ms windows to match the VAD decisions
    windows = tf.signal.frame(signal, window_frames, window_frames)
    # Filter signal with VAD decision == 1
    return tf.reshape(windows[vad_1], [-1])


''' We use wrappers here to store all the data for processing in one object '''
''' The dict object looks like so: {path, signal, sample_rate, logmelspec}'''
    
# loads in mp3 and stores signal and sample rate in the dict
def read_mp3_wrapper(x):
    # get the signal and rate from read_mp3
    signal, sample_rate = read_mp3(x["path"])
    print(signal)
    new_dict = dict(x, signal=signal, sample_rate=sample_rate)
    # dict now looks like: {path, signal, sample_rate}
    return new_dict

# removes silence from signal and stores in dict
def remove_silence_wrapper(x):
    new_dict = dict(x, signal=remove_silence(x["signal"], x["sample_rate"]))
    # dict now looks like: {path, signal, sample_rate}, now with removed silence on the signal
    return new_dict

# extracts features and stores in dict
def extract_features(x):
    with tf.device("GPU"):
        signal, rate = x["signal"], x["sample_rate"]
        logmelspecs = logmelspectrograms([signal], rate)
        logmelspecs_smn = cmvn(logmelspecs, normalize_variance=False)
    new_dict = dict(x, logmelspecs=logmelspecs_smn)
    # dict now looks like: {path, signal, sample_rate, logmelspecs}
    # where logmelspecs is the spectrogram we just created
    return new_dict

def signal_is_not_empty(x):
    return tf.size(x["signal"]) > 0

''' Prediction '''

# prediction function which takes an audio clip path and returns the highest confidence score from the model
# TODO implement signal_is_not_empty to remove signals which are empty
def predict(path, timing=False):
    ''' Audio Processing '''
    # timing variables, only used if timing is set to true, slightly inefficient, is ok
    before_audio = time.time()
    # read the file, remove silence, then get our spectrogram
    x = read_mp3_wrapper({"path": path})
    x = remove_silence_wrapper(x)
    x = extract_features(x)
    
    # our input data is in x['logmelspecs'][0]
    input_data = x['logmelspecs'][0]
    
    after_audio = time.time()
    ''' Resizing inputs '''
    before_resize = time.time()
    
    # we need to resize the input tensor on our model to match the size of the input_data
    # the input data has shape (num_frames, 40), need to extract that num_frames, which is index 0
    num_frames = input_data.shape[0]  
    # now we can resize the input tensor on the model, and reallocate the tensors
    interpreter.resize_tensor_input(input_details[0]['index'], (1, num_frames, 40))
    interpreter.allocate_tensors()
    # we also need to resize the input_data to match shape (1, num_frames, 40)
    input_data = np.reshape(input_data, (1, num_frames, 40))
    
    after_resize = time.time()
    ''' Inference '''
    before_inference = time.time()
    
    # now the sizes match we should be ready for inference
    # we start by setting the input tensor on the model to our input data
    interpreter.set_tensor(input_details[0]['index'], input_data)
    # now we can invoke it, which completes inference on the model
    interpreter.invoke()
    
    # the output is now located in the output details
    output_tensor = interpreter.get_tensor(output_details[0]['index'])
    # this is a 1d array containing the confidence scores for each langauge
    
    after_inference = time.time()

    # timing results
    if timing:
        time_audio      = '{:.2f}'.format(after_audio - before_audio)
        time_resize     = '{:.2f}'.format(after_resize - before_resize)
        time_inference  = '{:.2f}'.format(after_inference - before_inference)
        time_total      = '{:.2f}'.format(after_inference - before_audio)
        # lazy formatted string
        print(f"---TIMING---\n{time_audio}s Audio Processing \n{time_inference}s Inference \n{time_resize}s Resizing \n{time_total}s Total\n------------")
        
    # we can return the highest score as our final prediction
    prediction = output_tensor.argmax()
    return prediction

# language codes
languages = """
    et
    mn
    ta
    tr
""".split()
# language codes for mapping from index to language
target2lang = tuple(sorted(languages))

# model dir, change to saved tflite directory
model_dir = os.getcwd() + MODEL_PATH

# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(model_dir)
# input and output details contain information on the tensors for both input and output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# feel free to change this path to whatever file you wish to test
prediction = predict(AUDIO_PATH, timing=True) 
# finally we pass in the returned index into target2lang to map it to a language code
print(f"Predicted Language: {target2lang[prediction]}")