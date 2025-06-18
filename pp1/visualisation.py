import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from constants import *
from data_processing import midi_to_frame_matrices


def get_note_name(midi_number):
    """Convert MIDI note number to note name (e.g., 60 -> 'C4')."""
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number - 12) // 12
    note_idx = (midi_number - 12) % 12
    return f'{NOTES[note_idx]}{octave}'


def plot_midi_heatmap(predicted_notes):
    total_frames = predicted_notes.shape[0]
    total_time = (total_frames / FRAMES_IN_WINDOW) * WINDOW_TIME
    
    note_axis = np.arange(predicted_notes.shape[1])
    
    plt.figure(figsize=(12, 8))
    plt.imshow(predicted_notes.T, aspect='auto', origin='lower', extent=[0, total_time, note_axis[0], note_axis[-1]])
    
    plt.colorbar(label='Probability')
    plt.xlabel('Time (seconds)')
    plt.ylabel('MIDI Note Number')
    plt.title('Note Prediction Probabilities Over Time')
    
    plt.show()

    
def plot_midi_heatmaps(predicted_notes, predicted_onsets):
    total_frames = predicted_notes.shape[0]
    total_time = (total_frames / FRAMES_IN_WINDOW) * WINDOW_TIME
    note_axis = np.arange(predicted_notes.shape[1])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    im1 = ax1.imshow(predicted_notes.T, aspect='auto', origin='lower', 
                     extent=[0, total_time, note_axis[0], note_axis[-1]])
    plt.colorbar(im1, ax=ax1, label='Note Probability')
    ax1.set_ylabel('MIDI Note Number')
    ax1.set_title('Notes Over Time')
    
    # Plot onsets heatmap
    im2 = ax2.imshow(predicted_onsets.T, aspect='auto', origin='lower',
                     extent=[0, total_time, note_axis[0], note_axis[-1]])
    plt.colorbar(im2, ax=ax2, label='Onset Probability')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('MIDI Note Number')
    ax2.set_title('Onsets Over Time')
    
    plt.tight_layout()
    plt.show()

def plot_midi_heatmaps_with_actual_midi(predicted_notes, predicted_onsets, midi_path, save_path=None):
    total_frames = predicted_notes.shape[0]
    total_time = (total_frames / FRAMES_IN_WINDOW) * WINDOW_TIME
    note_axis = np.arange(predicted_notes.shape[1])
    time_points = np.linspace(0, total_time, total_frames)
    
    midi_matrices = midi_to_frame_matrices(midi_path, total_time * 2, WINDOW_TIME, FRAMES_IN_WINDOW)
    midi_notes = midi_matrices['note']
    
    first_note_frame = 0
    for i in range(midi_notes.shape[0]):
        if np.any(midi_notes[i]):
            first_note_frame = i
            break
    
    aligned_notes = midi_notes[first_note_frame:first_note_frame + total_frames]
    
    if aligned_notes.shape[0] < total_frames:
        padding = np.zeros((total_frames - aligned_notes.shape[0], aligned_notes.shape[1]))
        aligned_notes = np.vstack([aligned_notes, padding])
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    im1 = ax1.imshow(predicted_notes.T, aspect='auto', origin='lower', 
                     extent=[0, total_time, note_axis[0], note_axis[-1]])
    plt.colorbar(im1, ax=ax1, label='Note Probability')
    ax1.set_ylabel('MIDI Note Number')
    ax1.set_title('Predicted Notes Over Time')
    
    im2 = ax2.imshow(predicted_onsets.T, aspect='auto', origin='lower',
                     extent=[0, total_time, note_axis[0], note_axis[-1]])
    plt.colorbar(im2, ax=ax2, label='Onset Probability')
    ax2.set_ylabel('MIDI Note Number')
    ax2.set_title('Predicted Onsets Over Time')
    
    im3 = ax3.imshow(aligned_notes.T, aspect='auto', origin='lower',
                     extent=[0, total_time, note_axis[0], note_axis[-1]])
    plt.colorbar(im3, ax=ax3, label='Note Active')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('MIDI Note Number')
    ax3.set_title('Actual MIDI Notes')
    
    for note in range(aligned_notes.shape[1]):
        note_starts = np.where(np.diff(aligned_notes[:, note]) == 1)[0]
        
        note_starts = note_starts + 1
        
        onset_times = time_points[note_starts]
        
        for t in onset_times:
            ax1.axvline(x=t, color='white', linestyle='--', alpha=0.3, linewidth=1)
            ax2.axvline(x=t, color='white', linestyle='--', alpha=0.3, linewidth=1)
            ax3.axvline(x=t, color='white', linestyle='--', alpha=0.3, linewidth=1)
    
    midi_positions = []
    note_names = []
    
    first_c = MIN_MIDI
    while get_note_name(first_c)[0] != 'C':
        first_c += 1
    
    for midi_num in range(first_c, MAX_MIDI + 1, 12):
        note_name = get_note_name(midi_num)
        if note_name.startswith('C'):
            midi_positions.append(midi_num - MIN_MIDI)
            note_names.append(note_name)
    
    for ax in [ax1, ax2, ax3]:
        ax.set_yticks(midi_positions)
        ax.set_yticklabels(note_names)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def confidence_visusalisation(predicted_notes, predicted_onsets, save_path=None, confidence_threshold=0.5):
    total_frames = predicted_notes.shape[0]
    total_time = (total_frames / FRAMES_IN_WINDOW) * WINDOW_TIME
    note_axis = np.arange(predicted_notes.shape[1])
    
    thresholded_notes = (predicted_notes > confidence_threshold).astype(float)
    thresholded_onsets = (predicted_onsets > confidence_threshold).astype(float)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15), sharex=True)
    
    im1 = ax1.imshow(predicted_notes.T, aspect='auto', origin='lower', 
                     extent=[0, total_time, note_axis[0], note_axis[-1]])
    plt.colorbar(im1, ax=ax1, label='Note Probability')
    ax1.set_ylabel('MIDI Note Number')
    ax1.set_title('Raw Predicted Notes')
    
    im2 = ax2.imshow(thresholded_notes.T, aspect='auto', origin='lower',
                     extent=[0, total_time, note_axis[0], note_axis[-1]], 
                     cmap=im1.get_cmap())
    plt.colorbar(im2, ax=ax2, label=f'Notes > {confidence_threshold}')
    ax2.set_title('Thresholded Notes')
    
    im3 = ax3.imshow(predicted_onsets.T, aspect='auto', origin='lower',
                     extent=[0, total_time, note_axis[0], note_axis[-1]])
    plt.colorbar(im3, ax=ax3, label='Onset Probability')
    ax3.set_ylabel('MIDI Note Number')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_title('Raw Predicted Onsets')
    
    im4 = ax4.imshow(thresholded_onsets.T, aspect='auto', origin='lower',
                     extent=[0, total_time, note_axis[0], note_axis[-1]], 
                     cmap=im3.get_cmap())
    plt.colorbar(im4, ax=ax4, label=f'Onsets > {confidence_threshold}')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_title('Thresholded Onsets')
    
    midi_positions = []
    note_names = []
    
    first_c = MIN_MIDI
    while get_note_name(first_c)[0] != 'C':
        first_c += 1
    
    for midi_num in range(first_c, MAX_MIDI + 1, 12):
        note_name = get_note_name(midi_num)
        if note_name.startswith('C'):
            midi_positions.append(midi_num - MIN_MIDI)
            note_names.append(note_name)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_yticks(midi_positions)
        ax.set_yticklabels(note_names)
    
    plt.suptitle(f'Prediction Visualization (Confidence Threshold: {confidence_threshold})', 
                 fontsize=18, y=1.02)
    
    plt.tight_layout(rect=[0, 0, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()



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
