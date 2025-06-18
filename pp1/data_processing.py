import numpy as np
import pandas as pd
import tensorflow as tf
import pretty_midi
import librosa
import os

from constants import *


# ========================================= Generator =========================================

class WAVToMIDIDataGenerator(tf.keras.utils.Sequence):
	def __init__(self, df, dataset_path, batch_size, sr, window_time,
				overlap_time, frames_in_window, shuffle=True):
		self.df = df  # DataFrame with columns: audio_filename, midi_filename, duration
		self.dataset_path = dataset_path
		self.batch_size = batch_size
		self.sr = sr
		self.window_time = window_time
		self.overlap_time = overlap_time
		self.shuffle = shuffle
		self.cumulative_durations = [0] + df['duration'].cumsum().tolist()
		self.total_duration = self.df['duration'].sum()
		self.frames_in_window = frames_in_window

	def __len__(self):
		windows_count = count_windows(self.total_duration, self.window_time, self.overlap_time)
		
		# Number of batches that fit into the number of windows
		return int(np.ceil(windows_count / self.batch_size))

	def __getitem__(self, index):
		start_file_time, end_file_time, start_file_index, end_file_index = self._get_timings_and_file_indecies(index)

		audio_windows = []
		midi_windows_note = []
		midi_windows_onset = []
		
		for i in range(start_file_index, end_file_index + 1):
			file_duration = self.df.iloc[i]['duration']

			if i == start_file_index:
				start_time = start_file_time
			else:
				start_time = 0

			if i == end_file_index:
				end_time = end_file_time
			else:
				end_time = file_duration

			# AUDIO
			audio_path = os.path.join(self.dataset_path, self.df.iloc[i]['audio_filename'])
			midi_path = os.path.join(self.dataset_path, self.df.iloc[i]['midi_filename'])

			try:
				y, _ = librosa.load(audio_path, sr=self.sr, offset=start_time, duration=end_time - start_time)
				if len(y) == 0:
					print(f"Warning: Empty audio file {audio_path}")
					continue

				current_audio_windows = window_audio(y, self.sr, self.window_time, self.overlap_time)['windows']
				if not current_audio_windows:
					print(f"Warning: No windows created for {audio_path}")
					continue

				# MIDI
				midi_frames = midi_to_frame_matrices(midi_path, file_duration, self.window_time, self.frames_in_window)
				trimmed_midi_frames = trim_midi(midi_frames, self.window_time, self.frames_in_window, start_time=start_time, end_time=end_time)
				current_midi_windows = window_midi(trimmed_midi_frames, self.frames_in_window, self.window_time, self.overlap_time)

				audio_windows.extend(current_audio_windows)
				midi_windows_note.extend(current_midi_windows["note"])
				midi_windows_onset.extend(current_midi_windows["onset"])

			except Exception as e:
				print(f"Error processing file {audio_path}: {str(e)}")
				continue

		# Ensure we have at least one window
		if not audio_windows or not midi_windows_note or not midi_windows_onset:
			print(f"Warning: No valid windows for batch {index}")
			# Return a dummy batch with the correct shape
			dummy_audio = np.zeros((1, self.sr * self.window_time))
			dummy_midi = np.zeros((1, self.frames_in_window, PIANO_KEYS))
			return dummy_audio, {"note": dummy_midi, "onset": dummy_midi}

		# When taking more than 1 file, we can be faced with a higher number of windows than the batch size
		# In this case we need to cut the number of windows to the batch size
		audio_windows = audio_windows[:self.batch_size]
		midi_windows_note = midi_windows_note[:self.batch_size]
		midi_windows_onset = midi_windows_onset[:self.batch_size]

		X = tf.stack(audio_windows, axis=0)
		y_note = tf.stack(midi_windows_note, axis=0)
		y_onset = tf.stack(midi_windows_onset, axis=0)

		y = {"note": y_note, "onset": y_onset}
		return X, y

	def _get_timings_and_file_indecies(self, index):
		current_batch_start_time = windows_count_to_time(index * self.batch_size, self.window_time, self.overlap_time)
		current_batch_end_time = min(windows_count_to_time((index + 1) * self.batch_size, self.window_time, self.overlap_time), self.total_duration)

		start_file_index = self._get_file_index_for_time(current_batch_start_time)
		end_file_index = self._get_file_index_for_time(current_batch_end_time)

		start_file_time = self._get_file_time_from_file_index(current_batch_start_time, start_file_index)
		end_file_time = self._get_file_time_from_file_index(current_batch_end_time, end_file_index)

		return start_file_time, end_file_time, start_file_index, end_file_index

	def _get_file_index_for_time(self, time):
		for i in range(1, len(self.cumulative_durations)):
			if self.cumulative_durations[i] >= time:
				return i - 1 # File index
		return len(self.cumulative_durations) - 2 # Last file index

	def _get_file_time_from_file_index(self, current_batch_time, file_index):
		return current_batch_time - self.cumulative_durations[file_index] 

	def on_epoch_end(self):
		if self.shuffle:
			self.df = self.df.sample(frac=1).reset_index(drop=True)
			self.cumulative_durations = [0] + self.df['duration'].cumsum().tolist()

# ========================================= Windowing =========================================

# ----------------------------------------- AUDIO -----------------------------------------

def window_audio(y, sr, window_time, overlap_time):
	duration = len(y) / sr

	window_size = int(window_time * sr)
	hop_time = window_time - overlap_time
	hop_length = int(hop_time * sr)

	windows = []
	time_windows = []
		
	windows_count = count_windows(duration, window_time, overlap_time)
	duration_after_padding = windows_count_to_time(windows_count, window_time, overlap_time)
	amount_of_samples_to_pad = int((duration_after_padding * sr) - len(y)) # Padding with zeros to make the audio length a multiple of the window size

	if amount_of_samples_to_pad > 0:
		y_padded = np.pad(y, (0, amount_of_samples_to_pad), mode='constant', constant_values=0)
	else:
		y_padded = y

	for start in range(0, int(len(y_padded) - overlap_time * sr), hop_length):
		end = start + window_size
		window = y_padded[start:end]
		
		# HOTFIX: Sometimes the window is not exactly window_size in length, so we pad it with zeros once more
		if len(window) < window_size:
			window = np.pad(window, (0, window_size - len(window)), mode='constant', constant_values=0)
		
		windows.append(window)
		time_windows.append({
			"start": start / sr,
			"end": end / sr
		})

	return {
		"windows": windows, 
		"time_windows": time_windows, 
		"original_duration": duration, 
		"duration_after_padding": duration_after_padding
	}


# ------------------------------------------ MIDI -----------------------------------------

def window_midi(midi_frames, frames_in_window, window_time, overlap_time):
	fps = frames_in_window / window_time
	total_frames = midi_frames["note"].shape[0]
	overlap_frames = int(overlap_time * fps)
	hop_frames = frames_in_window - overlap_frames
	# total_frames_after_padding = int(np.ceil(total_frames / frames_in_window)) * frames_in_window
	total_frames_after_padding = int(np.ceil((total_frames - overlap_frames) / hop_frames)) * hop_frames + overlap_frames


	overlap_frames = int(overlap_time * fps)
	hop_frames = frames_in_window - overlap_frames

	notes = midi_frames["note"]
	onsets = midi_frames["onset"]

	amount_of_frames_to_pad = total_frames_after_padding - total_frames # Padding with zeros to make the audio length a multiple of the window size

	if amount_of_frames_to_pad > 0:
		notes = np.pad(midi_frames["note"], ((0, amount_of_frames_to_pad), (0, 0)), mode='constant', constant_values=0)
		onsets = np.pad(midi_frames["onset"], ((0, amount_of_frames_to_pad), (0, 0)), mode='constant', constant_values=0)
	
	note_windows = []
	onset_windows = []

	for start in range(0, int(notes.shape[0] - overlap_frames), hop_frames):
		note_matrix = notes[start:start+frames_in_window]
		onset_matrix = onsets[start:start+frames_in_window]
		note_windows.append(note_matrix)
		onset_windows.append(onset_matrix)

	windows = {"note": note_windows, "onset": onset_windows}

	return windows


# =========================================== Utils ===========================================

# ----------------------------------------- AUDIO -----------------------------------------

def count_windows(total_duration, window_time, overlap_time):
	_eps = 1e-9 # epsilon to avoid floating point errors
	
	# trivial or invalid cases
	if total_duration <= 0:
		return 0
	if window_time <= 0:
		raise ValueError("`window_time` must be positive.")
	step = window_time - overlap_time
	if step <= 0:
		raise ValueError("`overlap_time` must be smaller than `window_time`.")

	# one window is enough if the first already covers the signal
	if total_duration <= window_time + _eps:
		return 1

	return int(np.ceil((total_duration - window_time - _eps) / step)) + 1


def windows_count_to_time(window_count, window_time, overlap_time):
	if window_count <= 0:
		return 0.0
	hop_time = window_time - overlap_time
	return (window_count - 1) * hop_time + window_time


# ------------------------------------------ MIDI -----------------------------------------

def midi_to_frame_matrices(midi_path, duration, window_time, frames_in_window):
	fps = frames_in_window / window_time
	midi_data = pretty_midi.PrettyMIDI(midi_path)
	n_frames = int(np.ceil(duration * fps))
	
	note_matrix = np.zeros((n_frames, PIANO_KEYS))  # (time_samples, piano_keys)
	onset_matrix = np.zeros((n_frames, PIANO_KEYS))  # (time_samples, piano_keys)
	
	for instrument in midi_data.instruments:
		# In case there are multiple instruments, we will combine them into one matrix
		for note in instrument.notes:
			start_frame = int(note.start * fps)
			end_frame = int(note.end * fps) - 1 # -1 to avoid overlap of notes

			pitch_idx = note.pitch - MIN_MIDI  # MIN_MIDI = 21 = A0 (lowest piano key)
			
			# Check if note is inside of boundaries
			if start_frame >= n_frames or pitch_idx < 0 or pitch_idx >= 88:
					continue
			end_frame = min(end_frame, n_frames)

			note_matrix[start_frame:end_frame, pitch_idx] = 1.0
			onset_matrix[start_frame, pitch_idx] = 1.0
			
	return {"note": note_matrix, "onset": onset_matrix}


def trim_midi(midi_frames, window_time, frames_in_window, start_time=0, end_time=None):
	fps = frames_in_window / window_time
	
	start_frame = int(start_time * fps)
	if end_time is None or end_time > midi_frames["note"].shape[0]:
		end_frame = midi_frames["note"].shape[0]
	else:
		end_frame = int(end_time * fps)

	midi_frames["note"] = midi_frames["note"][start_frame:end_frame]
	midi_frames["onset"] = midi_frames["onset"][start_frame:end_frame]

	return midi_frames


# ========================================== Checks ===========================================

def test_generators(df):
	train_data = df[df['split'] == 'train']
	validation_data = df[df['split'] == 'validation']

	train_data = df[df['split'] == 'train']
	validation_data = df[df['split'] == 'validation']

	train_generator = WAVToMIDIDataGenerator(
		train_data, 
		dataset_path=DATASET_PATH,
		batch_size=BATCH_SIZE, 
		sr=SAMPLE_RATE, 
		window_time=WINDOW_TIME, 
		overlap_time=0, 
		frames_in_window=FRAMES_IN_WINDOW
	)

	validation_generator = WAVToMIDIDataGenerator(
		validation_data, 
		dataset_path=DATASET_PATH,
		batch_size=BATCH_SIZE, 
		sr=SAMPLE_RATE, 
		window_time=WINDOW_TIME, 
		overlap_time=0, 
		frames_in_window=FRAMES_IN_WINDOW
	)

	check_generator(train_generator, "train")
	check_generator(validation_generator, "validation")


def check_batch(batch_index, generator, name):
	X, y = generator[batch_index]
	print(f"Batch {batch_index+1}/{len(generator)}: {X.shape} {y['note'].shape} {y['onset'].shape}")
	
	# Add more detailed shape checks
	if X.shape[0] == 0 or y['note'].shape[0] == 0 or y['onset'].shape[0] == 0:
		print(f"ERROR: Zero-sized batch detected in {name} generator")
		print(f"X shape: {X.shape}")
		print(f"y['note'] shape: {y['note'].shape}")
		print(f"y['onset'] shape: {y['onset'].shape}")
		return False
		
	if X.shape[0] != y['note'].shape[0] or X.shape[0] != y['onset'].shape[0]:
		print(f"ERROR: Mismatched batch sizes in {name} generator")
		print(f"X shape: {X.shape}")
		print(f"y['note'] shape: {y['note'].shape}")
		print(f"y['onset'] shape: {y['onset'].shape}")
		return False
		
	# Check for NaN values
	if np.isnan(X).any() or np.isnan(y['note']).any() or np.isnan(y['onset']).any():
		print(f"ERROR: NaN values detected in {name} generator")
		print(f"X has NaN: {np.isnan(X).any()}")
		print(f"y['note'] has NaN: {np.isnan(y['note']).any()}")
		print(f"y['onset'] has NaN: {np.isnan(y['onset']).any()}")
		return False
		
	return True


def check_generator(generator, name):
	for i in range(len(generator)):
		if not check_batch(i, generator, name):
			break


# ============================================ Main ===========================================

if __name__ == "__main__":
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

	df = pd.read_csv(os.path.join(DATASET_PATH, "maestro-v3.0.0-processed.csv"))

	train_data = df[df['split'] == 'train']
	validation_data = df[df['split'] == 'validation']


	train_generator = WAVToMIDIDataGenerator(
		train_data, 
		dataset_path=DATASET_PATH,
		batch_size=BATCH_SIZE, 
		sr=SAMPLE_RATE, 
		window_time=WINDOW_TIME, 
		overlap_time=0, 
		frames_in_window=FRAMES_IN_WINDOW
	)

	validation_generator = WAVToMIDIDataGenerator(
		validation_data, 
		dataset_path=DATASET_PATH,
		batch_size=BATCH_SIZE, 
		sr=SAMPLE_RATE, 
		window_time=WINDOW_TIME, 
		overlap_time=0, 
		frames_in_window=FRAMES_IN_WINDOW
	)

	check_generator(train_generator, "train")
	check_generator(validation_generator, "validation")