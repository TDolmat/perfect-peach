import pandas as pd


batch_file_1 = '/Users/tomasz/Documents/Learning/Studies/zzMasters/Thesis/perfect-peach/pp1/saved_models/2506061601_model_epoch_batch_history.json'
batch_file_2 = '/Users/tomasz/Documents/Learning/Studies/zzMasters/Thesis/perfect-peach/pp1/saved_models/2506070036_model_epoch_02_batch_history.json'
batch_file_3 = '/Users/tomasz/Documents/Learning/Studies/zzMasters/Thesis/perfect-peach/pp1/saved_models/2506081136_model_epoch_02_08_batch_history.json'


df_1 = pd.read_json(batch_file_1)
df_2 = pd.read_json(batch_file_2)
df_3 = pd.read_json(batch_file_3)


max_epoch_1 = df_1['epoch'].max()
max_epoch_2 = df_2['epoch'].max()

df_2['epoch'] = df_2['epoch'] + max_epoch_1 + 1
df_3['epoch'] = df_3['epoch'] + max_epoch_1 + max_epoch_2 + 2

combined_df = pd.concat([df_1, df_2, df_3], ignore_index=True)

def calculate_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall + 1e-10)

combined_df['note_f1_score'] = calculate_f1(combined_df['note_precision'], combined_df['note_recall'])
combined_df['onset_f1_score'] = calculate_f1(combined_df['onset_precision'], combined_df['onset_recall'])

output_dir = '/Users/tomasz/Documents/Learning/Studies/zzMasters/Thesis/perfect-peach/pp1/histories/'
combined_df.to_csv(output_dir + 'batch_history_combined.csv', index=False)  