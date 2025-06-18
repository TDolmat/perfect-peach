import os
import librosa
import tensorflow as tf
import numpy as np
import argparse

from constants import *
from model import CQTLayer, HarmonicStackingLayer
from data_processing import window_audio
from visualisation import plot_midi_heatmaps, plot_midi_heatmaps_with_actual_midi, confidence_visusalisation


def predict(loaded_model_path, audio_path, overlap_time=0.3, duration=None, model=None):
    
    if model is None:
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


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    parser = argparse.ArgumentParser(description='Predict music transcription using trained model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file (.keras)')
    parser.add_argument('--audio_path', type=str, required=True, help='Path to the audio file to transcribe')
    parser.add_argument('--overlap_time', type=float, default=OVERLAP_PREDICTION_TIME, help='Overlap time between windows in seconds (default: 0.3)')
    parser.add_argument('--duration', type=float, default=None, help='Duration of audio to process in seconds (default: None, process entire file)')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save the prediction results (default: None, print to console)')
    parser.add_argument('--plot', action='store_true', default=True, help='Show plots of predicted notes and onsets (default: True)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Confidence threshold for visualization (default: 0.5)')
    
    args = parser.parse_args()
    
    predicted_notes, predicted_onsets = predict(
        loaded_model_path=args.model_path,
        audio_path=args.audio_path,
        overlap_time=args.overlap_time,
        duration=args.duration
    )
    
    print(f"Prediction completed!")
    
    if args.output_path:
        np.savez(args.output_path, notes=predicted_notes, onsets=predicted_onsets)
        print(f"Results saved to {args.output_path}")
    
    if args.plot:
        print("Generating visualization...")
        plot_midi_heatmaps(predicted_notes, predicted_onsets)
            