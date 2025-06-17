import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from constants import *


def visualize_cqt(C, title="Constant-Q Transform", note_separation=True, transpose=False, number_of_plots=1, tight_layout=False, y_label_interval=1, y_label_start_note=None):
	padding = 1

	if isinstance(C, tf.Tensor):
		C = C.numpy()

	if isinstance(C, np.ndarray):
		if len(C.shape) == 4 and C.shape[0] == 1 and C.shape[-1] > 1:
			C = [C[0, :, :, i] for i in range(C.shape[-1])]
			if not isinstance(title, (list, tuple)):
				title = [f"{title} - Channel {i+1}" for i in range(len(C))]
		elif len(C.shape) == 3 and C.shape[0] == 1:
			C = C[0]

	if isinstance(C, (list, tuple)):
		number_of_plots = len(C)
	
	fig, axes = plt.subplots(1, number_of_plots, figsize=(8*number_of_plots, 8), sharex=True, sharey=True)
	
	if number_of_plots == 1:
		axes = [axes]
	
	if not isinstance(C, (list, tuple)):
		C = [C]
		title = [title]
	elif not isinstance(title, (list, tuple)):
		title = [title] * len(C)

	for plot_idx in range(len(C)):
		current_C = C[plot_idx]
		
		if isinstance(current_C, tf.Tensor):
			current_C = current_C.numpy()
		
		if transpose:
			current_C = np.transpose(current_C)
			
		plt.sca(axes[plot_idx])
		
		if note_separation:
			C_padded = np.zeros((current_C.shape[0] + padding * PIANO_KEYS, current_C.shape[1]))
			
			for i in range(PIANO_KEYS):
				orig_start = i * BINS_PER_NOTE
				padded_start = i * (BINS_PER_NOTE + padding)
				if orig_start + BINS_PER_NOTE <= current_C.shape[0]:
					C_padded[padded_start:padded_start + BINS_PER_NOTE, :] = current_C[orig_start:orig_start + BINS_PER_NOTE, :]
			
			img = librosa.display.specshow(
				C_padded,
				sr=SAMPLE_RATE,
				x_axis='time',
				hop_length=CQT_HOP_LENGTH,
				bins_per_octave=12 * BINS_PER_NOTE,
				ax=axes[plot_idx]
			)
			
			if plot_idx == len(C) - 1:
				plt.colorbar(img, ax=axes[plot_idx], format='%+2.0f dB', aspect=50)
			
			note_positions = []
			note_labels = []
			
			for octave in OCTAVES:
				for note in NOTES:
					label = f'{note}{octave}'
					midi_note = librosa.note_to_midi(label)
					if MIN_MIDI <= midi_note <= MAX_MIDI:
						note_idx = midi_note - librosa.note_to_midi('A0')
						bin_idx = note_idx * (BINS_PER_NOTE + padding) + BINS_PER_NOTE // 2
						note_positions.append(bin_idx)
						note_labels.append(label)
			
			start_idx = 0
			if y_label_start_note:
				try:
					start_midi = librosa.note_to_midi(y_label_start_note)
					for i, label in enumerate(note_labels):
						if librosa.note_to_midi(label) >= start_midi:
							start_idx = i
							break
				except ValueError:
					print(f"Warning: Invalid note format for y_label_start_note: {y_label_start_note}")
			
			display_labels = [''] * len(note_labels)
			for i in range(start_idx, len(note_labels)):
				if (i - start_idx) % y_label_interval == 0:
					display_labels[i] = note_labels[i]
			
			plt.yticks(note_positions, display_labels)
			plt.grid(axis='y', linestyle='--', alpha=0.3)
		else:
			img = librosa.display.specshow(
				current_C,
				sr=SAMPLE_RATE,
				x_axis='time',
				y_axis='cqt_note',
				hop_length=CQT_HOP_LENGTH,
				bins_per_octave=12 * BINS_PER_NOTE,
				ax=axes[plot_idx]
			)
			
			if plot_idx == len(C) - 1:
				plt.colorbar(img, ax=axes[plot_idx], format='%+2.0f dB', pad=0.01, aspect=50)
		
		plt.title(title[plot_idx])
	
	plt.subplots_adjust(wspace=0.2)
	
	if tight_layout:
		plt.tight_layout()
	
	plt.show()
