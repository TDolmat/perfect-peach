import numpy as np

BATCH_SIZE = 64
EPOCHS = 20

# FILES
DATASET_PATH = '/Users/tomasz/Documents/Learning/Studies/zzMasters/Thesis/maestro-v3.0.0'
MODELS_PATH = '/Users/tomasz/Documents/Learning/Studies/zzMasters/Thesis/perfect-peach/pp1/saved_models'

# AUDIO
SAMPLE_RATE = 22050

# PIANO KEYS
PIANO_KEYS = 88
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
OCTAVES = list(range(0, 9))  # 0-8 octaves

# HORIZONTAL DIMENSION
# Windowing
WINDOW_TIME = 2 # 2 seconds
OVERLAP_PREDICTION_TIME = 0.3 # 0.3 seconds
OVERLAP_TRAINING_TIME = 0 # Overlap in training does not provide any benefit

# Constant-Q Transform
CQT_HOP_LENGTH = 256 # Audio samples in one frame
FRAMES_IN_WINDOW = int(np.ceil(WINDOW_TIME * SAMPLE_RATE / CQT_HOP_LENGTH)) # 173
FPS = FRAMES_IN_WINDOW / WINDOW_TIME # 173 / 2 = 86.5

# VERTICAL DIMENSION
BINS_PER_NOTE = 6 # Higher resolution: 6 bins per note
N_BINS = PIANO_KEYS * BINS_PER_NOTE # 528
BINS_PER_OCTAVE = 12 * BINS_PER_NOTE  # 72 (12 notes * bins per note)

# BOUNDARIES
FMIN = 27.5  # Lowest piano key frequency in Hz (A0)
MIN_MIDI = 21  # A0
MAX_MIDI = 108  # C8

HARMONICS = [0.5, 1, 2, 3, 4, 5, 6, 7]



# ================================ NOT IMPLEMENTED ================================

# LOSS
LABEL_SMOOTHING = 0.2
POSITIVE_WEIGHT = 0.5
NEGATIVE_WEIGHT = 0.5
NOTE_WEIGHT = 1.0
ONSET_WEIGHT = 1.5

# METRICS
POLYPHONY_THRESHOLD = 0.5