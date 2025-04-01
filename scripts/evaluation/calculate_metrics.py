import pandas as pd
import argparse, os, sys, math

from typing import Tuple

"""
parser = argparse.ArgumentParser()

parser.add_argument('results_fp', type=str, help='The filepath to the results CSV file to use in calculations.')
parser.add_argument('save_fp', type=str, help='The filepath to be used in saving the calculated metrics in a CSV file.')
parser.add_argument('data_type', type=int, choices=[0, 1], help='Integer representing the type of the input data (0 for validation, 1 for testing).')
parser.add_argument('predictions_colname', type=str, help='Name of column containing predicted classes.')
parser.add_argument('true_labels_colname', type=str, help='Name of column containing ground truth labels.')
parser.add_argument('file_colname', type=str, help='Name of column containing filepaths/filenames.')

parser.add_argument('--filepaths', action='store_true', help='Boolean flag indicating that the input dataset stores full filepaths instead of just filenames.')
parser.add_argument('--tracks', action='store_true', help='Boolean flag to calculate per-track metrics.')
parser.add_argument('--experiments', action='store_true', help='Boolean flag to calculate per-experiment metrics.')

args = parser.parse_args()
"""

# Hardcoded: Change per CSV
results_fp = "data/yolov11_testing.csv"
save_fp = "output/yolov11_testing.csv"
data_type = 1

# Hardcoded: Change per CSV
predictions_colname = "prediction"
true_labels_colname = "ground_truth"
file_colname = "filename"

# Hardcoded: Change per CSV
filepaths = False
tracks = False
experiments = True

if not os.path.exists(results_fp):
    print(f'Invalid Input: No file found at path {results_fp}!')
    sys.exit(1)

all_df = pd.read_csv(results_fp)

if filepaths:
    all_df[file_colname] = all_df[file_colname].apply(lambda x: x.split('/')[-1])

def compute_accuracies(df: pd.DataFrame) -> Tuple[float, float, float]:
    # Create boolean Series tracking correctness
    total_num_correct = (df[predictions_colname] == df[true_labels_colname]).sum()
    total_num_samples = df.shape[0]
    overall_acc = total_num_correct / total_num_samples

    # Use boolean masks for class 0 and class 1 labels
    cls0_mask = df[true_labels_colname] == 0
    cls1_mask = df[true_labels_colname] == 1

    # Compute class accuracies 
    cls0_num_correct = ((df[predictions_colname] == df[true_labels_colname]) & cls0_mask).sum()
    cls0_num_samples = cls0_mask.sum()
    cls0_acc = cls0_num_correct / cls0_num_samples if cls0_num_samples > 0 else 0.0

    cls1_num_correct = ((df[predictions_colname] == df[true_labels_colname]) & cls1_mask).sum()
    cls1_num_samples = cls1_mask.sum()
    cls1_acc = cls1_num_correct / cls1_num_samples if cls1_num_samples > 0 else 0.0

    return overall_acc, cls0_acc, cls1_acc

def compute_precisions_and_recalls(df: pd.DataFrame) -> Tuple[float, float, float, float]:
    cls0_tp = df[(df[predictions_colname] == 0) & (df[true_labels_colname] == 0)].shape[0]
    cls0_fp = df[((df[predictions_colname] == 0) & (df[true_labels_colname] == 1))].shape[0]
    cls0_fn = df[((df[predictions_colname] == 1) & (df[true_labels_colname] == 0))].shape[0]

    # FIXME: If zero set to 1?
    cls0_prec = cls0_tp / (cls0_tp + cls0_fp) if (cls0_tp + cls0_fp) > 0 else 1.0
    cls0_rec = cls0_tp / (cls0_tp + cls0_fn) if (cls0_tp + cls0_fn) > 0 else 1.0

    cls1_tp = df[(df[predictions_colname] == 1) & (df[true_labels_colname] == 1)].shape[0]
    cls1_fp = df[((df[predictions_colname] == 1) & (df[true_labels_colname] == 0))].shape[0]
    cls1_fn = df[((df[predictions_colname] == 0) & (df[true_labels_colname] == 1))].shape[0]

    # FIXME: See above comment
    cls1_prec = cls1_tp / (cls1_tp + cls1_fp) if (cls1_tp + cls1_fp) > 0 else 1.0
    cls1_rec = cls1_tp / (cls1_tp + cls1_fn) if (cls1_tp + cls1_fn) > 0 else 1.0

    return cls0_prec, cls0_rec, cls1_prec, cls1_rec

def compute_entropy(track_df: pd.DataFrame) -> float:
    cls0_count = track_df[track_df[predictions_colname] == 0].shape[0]
    cls1_count = track_df[track_df[predictions_colname] == 1].shape[0]

    track_length = track_df.shape[0]

    cls0_prob = cls0_count / track_length
    cls1_prob = cls1_count / track_length

    cls0_logprob = math.log(cls0_prob)
    cls1_logprob = math.log(cls1_prob)

    entropy = -(cls0_prob * cls0_logprob + cls1_prob * cls1_logprob)

    return entropy

data = []

all_rowname = 'all'
all_overall_acc, all_cls0_acc, all_cls1_acc = compute_accuracies(df=all_df)
all_cls0_prec, all_cls0_rec, all_cls1_prec, all_cls1_rec = compute_precisions_and_recalls(df=all_df)

all_record = [all_rowname, all_overall_acc, all_cls0_acc, all_cls1_acc, all_cls0_prec, all_cls0_rec, all_cls1_prec, all_cls1_rec, 'N/A']
data.append(all_record)

if experiments:
    all_df['experiment'] = all_df[file_colname].apply(lambda x: x.strip('__')[0])

    exp_names = all_df['experiment'].unique()
    for exp_name in exp_names:
        exp_df = all_df[all_df['experiment'] == exp_name] # CHANGE: Small typo

        exp_rowname = f'exp_{exp_name}'
        exp_overall_acc, exp_cls0_acc, exp_cls1_acc = compute_accuracies(df=exp_df)
        exp_cls0_prec, exp_cls0_rec, exp_cls1_prec, exp_cls1_rec = compute_precisions_and_recalls(df=exp_df)

        exp_record = [exp_rowname, exp_overall_acc, exp_cls0_acc, exp_cls1_acc, exp_cls0_prec, exp_cls0_rec, exp_cls1_prec, exp_cls1_rec, 'N/A']
        data.append(exp_record)

if tracks:
    all_df['track'] = all_df[file_colname].apply(lambda x: '__'.join(x.strip('__')[:-1]) if data_type == 0
                                                 else '_'.join(x.strip('_')[:-1]))
    
    trk_names = all_df['track'].unique()
    for trk_name in trk_names:
        trk_df = all_df[all_df['track'] == trk_name]

        trk_rowname = f'trk_{trk_name}'
        trk_overall_acc, trk_cls0_acc, trk_cls1_acc = compute_accuracies(df=trk_df)
        trk_cls0_prec, trk_cls0_rec, trk_cls1_prec, trk_cls1_rec = compute_precisions_and_recalls(df=trk_df)
        trk_entropy = compute_entropy(track_df=trk_df)

        trk_record = [trk_rowname, trk_overall_acc, trk_cls0_acc, trk_cls1_acc, trk_cls0_prec, trk_cls0_rec, trk_cls1_prec, trk_cls1_rec, trk_entropy]
        data.append(trk_record)

metrics_df = pd.DataFrame.from_records(data=data, columns=['name', 'overall_accuracy', 'class0_accuracy', 'class1_accuracy', 'class0_precision', 'class0_recall', 'class1_precision', 'class1_recall', 'entropy'])
metrics_df.to_csv(save_fp, index=False)