import json
import os
import pandas as pd
import tensorflow as tf
from datetime import datetime
import argparse

from constants import *
from data_processing import WAVToMIDIDataGenerator
from model import CQTLayer, HarmonicStackingLayer, oversimplified_model, pp1_model


class BatchHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.history = {'batch': [], 'epoch': []} 
        self.current_epoch = 0
        
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        self.history['batch'].append(batch)
        self.history['epoch'].append(self.current_epoch)

        for k, v in logs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(float(v))
    
    def on_epoch_end(self, epoch, logs=None):
        # Save after each epoch
        with open(self.filepath, 'w') as f:
            json.dump(self.history, f)
            
    def on_train_end(self, logs=None):
        # Final save at the end of training
        with open(self.filepath, 'w') as f:
            json.dump(self.history, f)


def train(load_path=None, save=False, epochs=4, full_dataset=True):
    df = pd.read_csv(os.path.join(DATASET_PATH, "maestro-v3.0.0-processed.csv"))

    if full_dataset:
        train_data = df[df['split'] == 'train']
        validation_data = df[df['split'] == 'validation']
    else:
        train_data = df[df['split'] == 'train'].iloc[0:1]
        validation_data = df[df['split'] == 'validation'].iloc[0:1]

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

    if load_path is None:
        model = pp1_model()
        file_name = 'model_epoch'
    else:
        print(f"Loading model from {load_path}")
        model = tf.keras.models.load_model(load_path, custom_objects={
            'CQTLayer': CQTLayer,
            'HarmonicStackingLayer': HarmonicStackingLayer
        })
        file_name = '_'.join(load_path.split('/')[-1].split('.')[0].split('_')[1:])
        

    model.compile(
        optimizer='adam',
        # loss=MusicTranscriptionLoss(),
        loss={
            'note': tf.keras.losses.BinaryCrossentropy(),
            'onset': tf.keras.losses.BinaryCrossentropy()
        },
        metrics={
            'note': [
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
            ],
            'onset': [
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
            ]
        }
    )

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        workers=6, 
        use_multiprocessing=True,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(MODELS_PATH, datetime.now().strftime("%y%m%d%H%M_") + file_name + '_{epoch:02d}.keras'),
                monitor='val_loss',
                save_best_only=False,
                save_weights_only=False,
                verbose=1
            ),
            BatchHistoryCallback(
                os.path.join(MODELS_PATH, datetime.now().strftime("%y%m%d%H%M_") + file_name + '_batch_history.json')
            )
        ] if save else []
    )



    
    if save:
        save_path = os.path.join(MODELS_PATH, datetime.now().strftime("%y%m%d%H%M_") + file_name + '_history.json')
        print(f"\nSaving training history to {save_path}")
        with open(save_path, 'w') as f:
            json.dump(history.history, f)
        
    return history


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    parser = argparse.ArgumentParser(description='Train PP1 model')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load a pre-trained model from')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs (default: 10)')
    parser.add_argument('--save', action='store_true', default=True, help='Save the model and training history (default: True)')
    parser.add_argument('--full_dataset', action='store_true', default=True, help='Use full dataset for training (default: True)')
    
    args = parser.parse_args()
    
    train(load_path=args.load_path, save=args.save, epochs=args.epochs, full_dataset=args.full_dataset)