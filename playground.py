import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '3' for even more silence
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt


# #!/usr/bin/env python
# # encoding: utf-8
# #
# # Copyright 2022 Spotify AB
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# from typing import Any, List

# import tensorflow as tf
# import tensorflow.keras.backend as K

# from basic_pitch.layers.math import log_base_b


# class HarmonicStacking(tf.keras.layers.Layer):
#     """Harmonic stacking layer

#     Input shape: (n_batch, n_times, n_freqs, 1)
#     Output shape: (n_batch, n_times, n_output_freqs, len(harmonics))

#     n_freqs should be much larger than n_output_freqs so that information from the upper
#     harmonics is captured.

#     Attributes:
#         bins_per_semitone: The number of bins per semitone of the input CQT
#         harmonics: List of harmonics to use. Should be positive numbers.
#         shifts: A list containing the number of bins to shift in frequency for each harmonic
#         n_output_freqs: The number of frequency bins in each harmonic layer.
#     """

#     def __init__(
#         self, bins_per_semitone: int, harmonics: List[float], n_output_freqs: int, name: str = "harmonic_stacking"
#     ):
#         """Downsample frequency by stride, upsample channels by 4."""
#         super().__init__(trainable=False, name=name)
#         self.bins_per_semitone = bins_per_semitone
#         self.harmonics = harmonics
#         self.shifts = [
#             int(tf.math.round(12.0 * self.bins_per_semitone * log_base_b(float(h), 2))) for h in self.harmonics
#         ]
#         self.n_output_freqs = n_output_freqs

#     def get_config(self) -> Any:
#         config = super().get_config().copy()
#         config.update(
#             {
#                 "bins_per_semitone": self.bins_per_semitone,
#                 "harmonics": self.harmonics,
#                 "n_output_freqs": self.n_output_freqs,
#                 "name": self.name,
#             }
#         )
#         return config

#     def call(self, x: tf.Tensor) -> tf.Tensor:
#         # (n_batch, n_times, n_freqs, 1)
#         tf.debugging.assert_equal(tf.shape(x).shape, 4)
#         channels = []
#         for shift in self.shifts:
#             if shift == 0:
#                 padded = x
#             elif shift > 0:
#                 paddings = tf.constant([[0, 0], [0, 0], [0, shift], [0, 0]])
#                 padded = tf.pad(x[:, :, shift:, :], paddings)
#             elif shift < 0:
#                 paddings = tf.constant([[0, 0], [0, 0], [-shift, 0], [0, 0]])
#                 padded = tf.pad(x[:, :, :shift, :], paddings)
#             else:
#                 raise ValueError

#             channels.append(padded)
#         x = tf.concat(channels, axis=-1)
#         x = x[:, :, : self.n_output_freqs, :]  # return only the first n_output_freqs frequency channels
#         return x


# class FlattenAudioCh(tf.keras.layers.Layer):
#     """Layer which removes a "channels" dimension of size 1.

#     Input shape: (batch, time, 1)
#     Output shape: (batch, time)
#     """

#     def call(self, x: tf.Tensor) -> tf.Tensor:
#         """x: (batch, time, ch)"""
#         shapes = K.int_shape(x)
#         tf.assert_equal(shapes[2], 1)
#         return tf.keras.layers.Reshape([shapes[1]])(x)  # ignore batch size


# class FlattenFreqCh(tf.keras.layers.Layer):
#     """Layer to flatten the frequency channel and make each channel
#     part of the frequency dimension.

#     Input shape: (batch, time, freq, ch)
#     Output shape: (batch, time, freq*ch)
#     """

#     def call(self, x: tf.Tensor) -> tf.Tensor:
#         shapes = K.int_shape(x)
#         return tf.keras.layers.Reshape([shapes[1], shapes[2] * shapes[3]])(x)  # ignore batch size

shifts = [1, 2, 3]
n_output_freqs = 528

def harmonic_stacking(x: tf.Tensor) -> tf.Tensor:
    channels = []
    for s in shifts:
        shift = int(np.log2(s) * 12 * BINS_PER_NOTE)
        if shift == 0:
            padded = x
        elif shift > 0:
            paddings = tf.constant([[0, 0], [0, 0], [0, shift], [0, 0]])
            padded = tf.pad(x[:, :, shift:, :], paddings)
        elif shift < 0:
            paddings = tf.constant([[0, 0], [0, 0], [-shift, 0], [0, 0]])
            padded = tf.pad(x[:, :, :shift, :], paddings)
        else:
            raise ValueError

        channels.append(padded)
    x = tf.concat(channels, axis=-1)
    x = x[:, :, : n_output_freqs, :]  # return only the first n_output_freqs frequency channels
    return x

# input_shape = (1, 173, 528, 1)
# x = tf.ones(input_shape)
# print(x.shape)

# harmonic_stacked_x = harmonic_stacking(x)

from constants import *
from model import cqt

file_path = "/Users/tomasz/Documents/Learning/Studies/zzMasters/Thesis/test_tracks/c_major.wav"
y, sr = librosa.load(file_path, duration=WINDOW_TIME, sr=SAMPLE_RATE, offset=0.5)
print(y.shape)
C = cqt(y)
print(C.shape)


C = C.reshape(1, 173, 528, 1)
print(C.shape)
harmonic_stacked_C = harmonic_stacking(C)
print(harmonic_stacked_C.shape)

def visualize_harmonic_stacking(x: tf.Tensor):
    # Convert TensorFlow tensor to numpy array
    x_np = x.numpy()
    
    fig, ax = plt.subplots(3,1, figsize=(12, 6))
    
    img1 = librosa.display.specshow(x_np[0,:,:,0].T, sr=SAMPLE_RATE, x_axis='time', y_axis='cqt_note', ax=ax[0])
    ax[0].set_title('Constant-Q Transform 1')
    fig.colorbar(img1, ax=ax[0], format='%+2.0f dB')
    
    img2 = librosa.display.specshow(x_np[0,:,:,1].T, sr=SAMPLE_RATE, x_axis='time', y_axis='cqt_note', ax=ax[1])
    ax[1].set_title('Constant-Q Transform 2')
    fig.colorbar(img2, ax=ax[1], format='%+2.0f dB')
    
    img3 = librosa.display.specshow(x_np[0,:,:,2].T, sr=SAMPLE_RATE, x_axis='time', y_axis='cqt_note', ax=ax[2])
    ax[2].set_title('Constant-Q Transform 3')
    fig.colorbar(img3, ax=ax[2], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.show()
    
