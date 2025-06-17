import librosa
import warnings
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, Reshape, Concatenate

from matplotlib import pyplot as plt
import librosa.display

from constants import *
from visualisation import visualize_cqt


# =========================================== Model ===========================================


def pp1_model():
    inputs = tf.keras.Input(shape=(SAMPLE_RATE * WINDOW_TIME,))  # (BATCH_SIZE, 44100)
    # ------------------------------ Audio transformations ------------------------------
    x = CQTLayer()(inputs) # (BATCH_SIZE, 173, 528)
    x = HarmonicStackingLayer(harmonics=HARMONICS)(x) # (BATCH_SIZE, 173, 528, 8)

    # ---------------------------------- Note detection ---------------------------------
    x_notes = Conv2D(
        filters=8,
        kernel_size=(3, BINS_PER_OCTAVE), # (3, 72)
        padding="same",
        name="note_conv1"
    )(x) # (BATCH_SIZE, 173, 528, 8)
    x_notes = BatchNormalization()(x_notes)
    x_notes = ReLU()(x_notes)

    x_notes = Conv2D(
        filters=32,
        kernel_size=(6, 6),
        strides=(1, 2),
        padding="same",
        name="note_conv2"
    )(x_notes) # (BATCH_SIZE, 173, 264, 32)
    x_notes = BatchNormalization()(x_notes)
    x_notes = ReLU()(x_notes)

    x_notes = Conv2D(
        filters=1,
        kernel_size=(6, 3),
        strides=(1, 3),
        padding="same",
        activation="sigmoid",
        name="note_prediction"
    )(x_notes) # (BATCH_SIZE, 173, 88, 1)  

    x_notes_output = Reshape(
        target_shape=(FRAMES_IN_WINDOW, PIANO_KEYS), 
        name="note"
    )(x_notes) # (BATCH_SIZE, 173, 88)

    # --------------------------------- Onset detection ---------------------------------
    x_onset = Conv2D(
        filters=32,
        kernel_size=(6, 6),
        strides=(1, 6),
        padding="same",
        name="onset_conv1"
    )(x) # (BATCH_SIZE, 173, 88, 32)
    x_onset = BatchNormalization()(x_onset)
    x_onset = ReLU()(x_onset)

    x_onset = Concatenate(axis=-1)([x_notes, x_onset]) # (BATCH_SIZE, 173, 88, 33)

    x_onset = Conv2D(
        filters=1,
        kernel_size=(6, 6),
        padding="same",
        activation="sigmoid",
    )(x_onset) # (BATCH_SIZE, 173, 88, 1)

    x_onset_output = Reshape(
        target_shape=(FRAMES_IN_WINDOW, PIANO_KEYS), 
        name="onset"
    )(x_onset) # (BATCH_SIZE, 173, 88)
    
    outputs = {"onset": x_onset_output, "note": x_notes_output}
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def oversimplified_model():
    inputs = tf.keras.Input(shape=(SAMPLE_RATE * WINDOW_TIME,))  # (BATCH_SIZE, 44100)
    x = CQTLayer()(inputs) # (BATCH_SIZE, 173, 528)
    x = HarmonicStackingLayer(harmonics=HARMONICS)(x) # (BATCH_SIZE, 173, 528, 8)
    x = Conv2D(
        filters=1,
        kernel_size=(6, 6),
        strides=(1, 6),
        padding="same",
        activation="sigmoid",
    )(x) # (BATCH_SIZE, 173, 88, 1)
    x = Reshape(target_shape=(FRAMES_IN_WINDOW, PIANO_KEYS))(x) # (BATCH_SIZE, 173, 88)
    return tf.keras.Model(inputs=inputs, outputs={"onset": x, "note": x})

# ========================================== Layers ===========================================


class CQTLayer(Layer):
    """
    Applies a Constant-Q Transform to the input audio.
    Transposes the output to (BATCH_SIZE, 173, 528) to match the expected output shape of the model.
    """
    def __init__(self, **kwargs):
        super(CQTLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        # Use tf.py_function to wrap the cqt function
        cqt_output = tf.py_function(func=cqt, inp=[inputs], Tout=tf.float32)
        
        # (BATCH_SIZE, 528, 173) -> (BATCH_SIZE, 173, 528)
        cqt_output = tf.transpose(cqt_output, perm=[0, 2, 1])
        
        # For model summary - ensure the output shape is correct
        cqt_output = tf.reshape(cqt_output, [-1, FRAMES_IN_WINDOW, N_BINS])
        
        return cqt_output


class HarmonicStackingLayer(Layer):
    """
    Applies harmonic stacking to the input CQT.
    Shifts the CQT by the number of bins specified in the harmonics list.
    For the positive harmonics, the CQT is shifted down by the number of bins specified in the harmonics list.
    For the negative harmonics, the CQT is shifted up by the number of bins specified in the harmonics list.
    The CQT is then concatenated along the last dimension.
    The output shape is (BATCH_SIZE, 173, 528, len(harmonics)).
    """
    def __init__(self, harmonics: list[float], **kwargs):
        super(HarmonicStackingLayer, self).__init__(**kwargs)
        self.harmonics = harmonics
    
    def call(self, inputs):
        return harmonic_stacking(inputs, self.harmonics)


# =========================================== Utils ===========================================


def cqt(y):
    # Convert TensorFlow tensor to NumPy array
    if isinstance(y, tf.Tensor):
        y = y.numpy()
    
    with warnings.catch_warnings():
        # Ignore warnings
        warnings.filterwarnings("ignore", message="n_fft=.* is too large for input signal")

        C = librosa.cqt(
            y,
            sr=SAMPLE_RATE,
            hop_length=CQT_HOP_LENGTH,
            n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE,
            fmin=FMIN
        )

        C_db = librosa.power_to_db(np.abs(C), ref=np.max)
    return C_db


def harmonic_stacking(x: tf.Tensor, harmonics: list[int], n_output_freqs:int = N_BINS):
    if len(x.shape) == 3:
        # from (BATCH_SIZE, 173, 528) to (BATCH_SIZE, 173, 528, 1) [the channel dimension]
        x = tf.expand_dims(x, axis=-1)

    channels = []

    for harmonic in harmonics:
        shift = int(np.log2(harmonic) * BINS_PER_OCTAVE)
        
        if shift == 0:
            x_padded = x
        elif shift > 0:
            # Shift "down" by the number of shifted bins
            x_shifted = x[:, :, shift:, :]
            x_padded = tf.pad(x_shifted, [[0, 0], [0, 0], [0, shift], [0, 0]])
        elif shift < 0:
            # Shift "up" by the number of shifted bins
            x_shifted = x[:, :, :shift, :]
            x_padded = tf.pad(x_shifted, [[0, 0], [0, 0], [-shift, 0], [0, 0]])

        channels.append(x_padded)
    x_concat = tf.concat(channels, axis=-1)
    return x_concat[:, :, : n_output_freqs, :]




def test1():
    file_path = "/Users/tomasz/Documents/Learning/Studies/zzMasters/Thesis/test_tracks/c_major.wav"
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    print(y.shape)
    y = y.reshape(1, -1)
    print(y.shape)



    # model = tf.keras.Sequential([
    #     CQTLayer(),
    # ])

    cqt_output = cqt(y)
    visualize_cqt(cqt_output, transpose=False)

def test_visualisation():
    wav_path = '/Users/tomasz/Documents/Learning/Studies/zzMasters/Thesis/test_tracks/c_chord.wav'
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, duration=WINDOW_TIME)
    
    y = y.reshape(1, -1)

    cqt_output = cqt(y)
    visualize_cqt(cqt_output, transpose=False, title="CQT spectrogram", note_separation=True, number_of_plots=1, tight_layout=True)

def test_visualisation2():
    wav_path = '/Users/tomasz/Documents/Learning/Studies/zzMasters/Thesis/test_tracks/c_major.wav'
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    
    y = y.reshape(1, -1)

    cqt_output = cqt(y)
    visualize_cqt(cqt_output, transpose=False, title="Constant-Q Transform spectrogram", note_separation=True, number_of_plots=1, tight_layout=True)

def test2():
    file_path = "/Users/tomasz/Documents/Learning/Studies/zzMasters/Thesis/test_tracks/c_chord.wav"
    y, sr = librosa.load(file_path, duration=WINDOW_TIME, sr=SAMPLE_RATE, offset=0)
    print(y.shape)
    y = y.reshape(1, -1)
    print(y.shape)

    model = tf.keras.Sequential([
        CQTLayer(),
        HarmonicStackingLayer(harmonics=[0.5, 1, 2, 3]),
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    input_shape = SAMPLE_RATE * WINDOW_TIME  # This should be an integer
    model.build(input_shape=(None, input_shape))  # Pass as a single tuple
    model.summary()

    prediction = model.predict(y)

    # visualize_cqt(prediction, transpose=True, number_of_plots=3, y_label_interval=2, y_label_start_note="A#0")
    visualize_cqt(prediction, transpose=True, number_of_plots=1, y_label_interval=2, y_label_start_note="A#0")


def test3():
    model = pp1_model()
    model.summary()


def plot():
    model = pp1_model()
    from keras.utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

def summary():
    model = pp1_model()
    model.summary()


if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '3' for even more silence
    # test1()
    test2()
    # test3()
    # plot()
    # summary()
    # test_visualisation()
    # test_visualisation2()



def get_metrics():
    return 
