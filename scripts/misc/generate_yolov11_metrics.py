import pandas as pd

# Derive ground-truth class from filepath
def extract_true_class(path):
    path_lower = path.lower()
    if 'female' in path_lower:
        return 'female'
    elif 'male' in path_lower:
        return 'male'

# Load the matched predictions file
df = pd.read_csv('pred_gt_matched.csv')

# Create ground truths based on folder image came from (from test set)
df['true_class'] = df['filepath'].apply(extract_true_class)

# Count overall correct guesses and then overall accuracy as percentage
df['correct'] = (df['class_name'] == df['true_class'])
overall_accuracy = df['correct'].mean() * 100

# Prepare per-class metrics
classes = ['male', 'female']
metrics = []
metrics = {}
for cls in classes:
    # Compute true positives, false positives, and false negatives
    tp = ((df['class_name'] == cls) & (df['true_class'] == cls)).sum()
    fp = ((df['class_name'] == cls) & (df['true_class'] != cls)).sum()
    fn = ((df['class_name'] != cls) & (df['true_class'] == cls)).sum()
    total_true = (df['true_class'] == cls).sum()
    
    # Compute class accuracy, class precision, and class recall
    class_accuracy = (tp / total_true * 100) if total_true > 0 else 0.0
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0

    # Add to dict for later printing out
    metrics[cls] = {
        'accuracy': class_accuracy,
        'precision': precision,
        'recall': recall
    }


# Print results
print(f"Overall accuracy: {overall_accuracy:.2f}%\n")
for cls in classes:
    m = metrics[cls]
    print(f"{cls.capitalize()} accuracy: {m['accuracy']:.2f}%")
    print(f"{cls.capitalize()} precision: {m['precision']:.2f}%")
    print(f"{cls.capitalize()} recall: {m['recall']:.2f}%")
