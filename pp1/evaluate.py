import os
import pandas as pd
import tensorflow as tf
import json
from datetime import datetime

from constants import *
from data_processing import WAVToMIDIDataGenerator
from model import CQTLayer, HarmonicStackingLayer

def evaluate_model(model_path):
    df = pd.read_csv(os.path.join(DATASET_PATH, "maestro-v3.0.0-processed.csv"))

    test_data = df[df['split'] == 'test']

    test_generator = WAVToMIDIDataGenerator(
        test_data, 
        dataset_path=DATASET_PATH,
        batch_size=BATCH_SIZE, 
        sr=SAMPLE_RATE, 
        window_time=WINDOW_TIME, 
        overlap_time=0, 
        frames_in_window=FRAMES_IN_WINDOW
    )
    
    model = tf.keras.models.load_model(model_path, custom_objects={
        'CQTLayer': CQTLayer,
        'HarmonicStackingLayer': HarmonicStackingLayer,
    })
    
    print(f"\nEvaluating model: {model_path}")
    results = model.evaluate(
        test_generator,
        verbose=1,
        return_dict=True
    )
    
    print("\nEvaluation Results:")
    print("-" * 50)
    for metric_name, value in results.items():
        print(f"{metric_name}: {value:.4f}")
    
    results_dir = os.path.join(os.path.dirname(model_path), 'evaluations')
    os.makedirs(results_dir, exist_ok=True)
    
    evaluation_results = {
        'model_path': model_path,
        'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'test_set_size': len(test_data),
        'metrics': {k: float(v) for k, v in results.items()} 
    }
    
    model_name = os.path.basename(model_path)
    timestamp = datetime.now().strftime("%y%m%d%H%M")
    results_filename = f"{model_name.split('.')[0]}_evaluation_{timestamp}.json"
    results_path = os.path.join(results_dir, results_filename)
    
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    
    print(f"\nEvaluation results saved to: {results_path}")
    
    return results

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    model_path = "/Users/tomasz/Documents/Learning/Studies/zzMasters/Thesis/perfect-peach/pp1/saved_models/2506081136_model_epoch_02_08_10.keras"
    evaluate_model(model_path)

    # ========================================== Results ==========================================

    # loss = 0.04343153163790703
    # note_loss = 0.03929929435253143
    # onset_loss = 0.004132249858230352
    # note_accuracy = 0.29429900646209717
    # note_note_precision = 0.7365725636482239
    # note_note_recall = 0.41383132338523865
    # onset_accuracy = 0.0597008652985096
    # onset_onset_precision = 0.48809731006622314
    # onset_onset_recall = 0.25760218501091003


    # note_f1_score = 2 * (note_note_precision * note_note_recall) / (note_note_precision + note_note_recall)
    # onset_f1_score = 2 * (onset_onset_precision * onset_onset_recall) / (onset_onset_precision + onset_onset_recall)
    # print(note_f1_score)
    # print(onset_f1_score)