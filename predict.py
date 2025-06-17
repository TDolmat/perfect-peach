import os
import librosa
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from constants import *
from model import CQTLayer, HarmonicStackingLayer
from data_processing import midi_to_frame_matrices, window_audio

def get_note_name(midi_number):
    """Convert MIDI note number to note name (e.g., 60 -> 'C4')."""
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number - 12) // 12
    note_idx = (midi_number - 12) % 12
    return f'{NOTES[note_idx]}{octave}'

def predict(loaded_model_path, audio_path, overlap_time=0.3, duration=None):
    
    model = tf.keras.models.load_model(loaded_model_path, custom_objects={
        'CQTLayer': CQTLayer,
        'HarmonicStackingLayer': HarmonicStackingLayer
    })

    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, duration=duration)
    
    original_duration = len(y) / SAMPLE_RATE
    original_frames = int(np.ceil(original_duration * FPS))
    
    windows = window_audio(y, sr=SAMPLE_RATE, window_time=WINDOW_TIME, overlap_time=overlap_time)

    print(f"Original duration: {original_duration:.2f} seconds ({original_frames} frames)")
    print(f"Number of windows: {len(windows['windows'])}")

    predictions = []
    print(len(windows['windows']))
    for window in windows['windows']:
        window = window.reshape(1, -1)
        pred = model.predict(window)
        predictions.append(pred)

    predicted_notes = []
    predicted_onsets = []

    for pred in predictions:
        predicted_notes.append(pred['note'][0])
        predicted_onsets.append(pred['onset'][0])

    overlap_frames = int(np.ceil(FPS * overlap_time))
    non_overlap_frames = FRAMES_IN_WINDOW - overlap_frames

    total_frames = (len(predictions) - 1) * non_overlap_frames + FRAMES_IN_WINDOW
    final_notes = np.zeros((total_frames, predicted_notes[0].shape[1]))
    final_onsets = np.zeros((total_frames, predicted_onsets[0].shape[1]))

    final_notes[:FRAMES_IN_WINDOW] = predicted_notes[0]
    final_onsets[:FRAMES_IN_WINDOW] = predicted_onsets[0]

    for i in range(1, len(predictions)):
        current_start = i * non_overlap_frames
        current_end = current_start + FRAMES_IN_WINDOW
        
        overlap_start = current_start
        overlap_end = current_start + overlap_frames
        
        final_notes[overlap_start:overlap_end] = (
            final_notes[overlap_start:overlap_end] + predicted_notes[i][:overlap_frames]
        ) / 2
        final_onsets[overlap_start:overlap_end] = (
            final_onsets[overlap_start:overlap_end] + predicted_onsets[i][:overlap_frames]
        ) / 2
        
        final_notes[overlap_end:current_end] = predicted_notes[i][overlap_frames:]
        final_onsets[overlap_end:current_end] = predicted_onsets[i][overlap_frames:]

    predicted_notes = final_notes[:original_frames]
    predicted_onsets = final_onsets[:original_frames]
    
    return predicted_notes, predicted_onsets


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

def confidence_vis(predicted_notes, predicted_onsets, midi_path, save_path=None, confidence_threshold=0.5):
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



if __name__ == "__main__":
    # loaded_model_path = '/Users/tomasz/Documents/Learning/Studies/zzMasters/Thesis/perfect-peach/pp1/saved_models/first_tests/2506050720_model_epoch_01_01.h5'
    loaded_model_path = '/Users/tomasz/Documents/Learning/Studies/zzMasters/Thesis/perfect-peach/pp1/saved_models/2506081136_model_epoch_02_08_10.keras'
    # audio_path = '/Users/tomasz/Documents/Learning/Studies/zzMasters/Thesis/test_tracks/c_chord.wav'
    audio_path = '/Users/tomasz/Downloads/for_elise_by_beethoven.wav'
    
    # midi_path = '/Users/tomasz/Documents/Learning/Studies/zzMasters/Thesis/maestro-v3.0.0/2018/MIDI-Unprocessed_Recital13-15_MID--AUDIO_14_R1_2018_wav--3.midi'
    midi_path = '/Users/tomasz/Downloads/for_elise_by_beethoven.mid'

    predicted_notes, predicted_onsets = predict(loaded_model_path, audio_path, overlap_time=OVERLAP_PREDICTION_TIME, duration=5)
    print(predicted_notes.shape)
    print(predicted_onsets.shape)
    # plot_midi_heatmaps(predicted_notes, predicted_onsets)
    # save_path = '/Users/tomasz/Documents/Learning/Studies/zzMasters/Thesis/perfect-peach/pp1/results/plots/for_elise_by_beethoven.png'
    save_path = '/Users/tomasz/Documents/Learning/Studies/zzMasters/Thesis/perfect-peach/pp1/results/plots/confidence.png'
    # save_path = None


    # plot_midi_heatmaps_with_actual_midi(predicted_notes, predicted_onsets, midi_path, save_path=save_path)

    confidence_vis(predicted_notes, predicted_onsets, midi_path, save_path=save_path)

    # midi_matrices = midi_to_frame_matrices(midi_path, 4, WINDOW_TIME, FRAMES_IN_WINDOW)

    # print(midi_matrices['note'])
    # plot_midi_heatmaps(midi_matrices['note'], midi_matrices['onset'])
    

