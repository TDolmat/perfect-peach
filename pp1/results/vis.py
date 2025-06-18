import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df_epoch = pd.read_csv('/Users/tomasz/Documents/Learning/Studies/zzMasters/Thesis/perfect-peach/pp1/histories/epoch_history_combined.csv')
df_batch = pd.read_csv('/Users/tomasz/Documents/Learning/Studies/zzMasters/Thesis/perfect-peach/pp1/histories/batch_history_combined.csv')

def clean_and_smooth(series, window_size=100, std_threshold=2.5):
    rolling_mean = series.rolling(window=window_size, center=True).mean()
    rolling_std = series.rolling(window=window_size, center=True).std()
    
    lower_bound = rolling_mean - std_threshold * rolling_std
    upper_bound = rolling_mean + std_threshold * rolling_std
    
    clean_series = series.copy()
    clean_series[(series < lower_bound) | (series > upper_bound)] = np.nan
    clean_series = clean_series.interpolate(method='linear')
    
    smooth_series = clean_series.rolling(window=window_size, center=True).mean()
    
    final_smooth = smooth_series.rolling(window=window_size//2, center=True).mean()
    
    return final_smooth

def plot_metric_grid(df_epoch, df_batch, metric_key, batches_per_epoch=4491, mode='SHOW'):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 16))
    
    fig.suptitle(f'Training {metric_key.replace("_", " ").title()} Analysis', 
                 fontsize=20, y=0.96)

    epoch_indices = np.arange(1, len(df_epoch) + 1)
    batch_epochs = df_batch.index / batches_per_epoch

    note_col = f'note_{metric_key}'
    onset_col = f'onset_{metric_key}'
    note_val_col = f'val_note_{metric_key}'
    onset_val_col = f'val_onset_{metric_key}'

    metric_name = metric_key.replace("_", " ").title() if metric_key != 'f1_score' else 'F1-score'

    clean_note = clean_and_smooth(df_batch[note_col])
    clean_onset = clean_and_smooth(df_batch[onset_col])

    ax1.plot(epoch_indices, df_epoch[note_col], 'b-', label='Epoch', linewidth=2)
    ax1.plot(batch_epochs, clean_note, 'b--', alpha=0.5, label='Batch (smoothed)', linewidth=1)
    ax1.set_title(f'Note {metric_name}', pad=20, fontsize=16)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(metric_name)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xticks(np.arange(0, 21))
    ax1.set_xlim(0, 20)

    ax2.plot(epoch_indices, df_epoch[onset_col], 'r-', label='Epoch', linewidth=2)
    ax2.plot(batch_epochs, clean_onset, 'r--', alpha=0.5, label='Batch (smoothed)', linewidth=1)
    ax2.set_title(f'Onset {metric_name}', pad=20, fontsize=16)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(metric_name)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xticks(np.arange(0, 21))
    ax2.set_xlim(0, 20)

    if note_val_col in df_epoch.columns:
        ax3.plot(epoch_indices, df_epoch[note_val_col], color='#2ecc71', label='Epoch', linewidth=2)
    ax3.set_title(f'Note Validation {metric_name}', pad=20, fontsize=16)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel(metric_name)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xticks(np.arange(0, 21))
    ax3.set_xlim(0, 20)

    if onset_val_col in df_epoch.columns:
        ax4.plot(epoch_indices, df_epoch[onset_val_col], color='#f39c12', label='Epoch', linewidth=2)
    ax4.set_title(f'Onset Validation {metric_name}', pad=20, fontsize=16)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel(metric_name)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xticks(np.arange(0, 21))
    ax4.set_xlim(0, 20)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95], h_pad=4.0, w_pad=2.0)

    if mode == 'SAVE':
        plt.savefig(f'/Users/tomasz/Documents/Learning/Studies/zzMasters/Thesis/perfect-peach/pp1/results/plots/{metric_key}_analysis.png', 
                    dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    total_batches = len(df_batch)
    total_epochs = len(df_epoch)
    batches_per_epoch = total_batches // total_epochs
    
    print(f"Total batches: {total_batches}")
    print(f"Total epochs: {total_epochs}")
    print(f"Batches per epoch: {batches_per_epoch}")

    # mode = 'SAVE'
    mode = 'SHOW'
    
    plot_metric_grid(df_epoch, df_batch, 'loss', batches_per_epoch)
    plot_metric_grid(df_epoch, df_batch, 'accuracy', batches_per_epoch, mode)
    plot_metric_grid(df_epoch, df_batch, 'precision', batches_per_epoch)
    plot_metric_grid(df_epoch, df_batch, 'recall', batches_per_epoch)