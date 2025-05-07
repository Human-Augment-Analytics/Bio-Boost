import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv


def generate_radial_plot(model1_path, model2_path, metric_col="overall_accuracy", figsize=(12, 9), title=None, image_name="temp"):
    # Read first model's data
    df1 = pd.read_csv(model1_path)
    model1_filtered = df1[df1['name'].str.contains('exp', case=False)]  # filter out tracks
    
    # Read second model's data
    df2 = pd.read_csv(model2_path)
    model2_filtered = df2[df2['name'].str.contains('exp', case=False)]  # filter out tracks
    
    
    # Compare only the same experiments between two models
    common_experiments = list(set(model1_filtered['name']).intersection(
                             set(model2_filtered['name'])))
    print(f"Number of common experiments among models: {len(common_experiments)}")

    # Filter each model's data to have same experiments
    model1_filtered = model1_filtered[model1_filtered['name'].isin(common_experiments)]
    model2_filtered = model2_filtered[model2_filtered['name'].isin(common_experiments)]
    
    # Sort to ensure alignment - the index of experiment needs to match the index of accuracy
    model1_filtered = model1_filtered.sort_values('name')
    model2_filtered = model2_filtered.sort_values('name')

    # Get experiment names and accuracy values
    experiment_names = model1_filtered['name'].tolist()
    model1_accuracy = model1_filtered[metric_col].tolist()
    model2_accuracy = model2_filtered[metric_col].tolist()
    
    # Print debugging
    print(f"Model 1 Accuracy Exp: {len(model1_accuracy)}")
    print(f"Model 2 Accuracy Exp: {len(model2_accuracy)}")
    
    # Create simplified naming conventions
    simplified_names = [f"exp{i+1}" for i in range(len(common_experiments))]

    # Calculate the angle for each experiment
    angles = np.linspace(0, 2*np.pi, len(simplified_names), endpoint=False).tolist()

    # For circular plot
    simplified_names.append(simplified_names[0])
    model1_accuracy.append(model1_accuracy[0])
    model2_accuracy.append(model2_accuracy[0])
    angles.append(angles[0])

    # Create the plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)

    # Plot the accuracy values for models with different colors
    ax.plot(angles, model1_accuracy, '-', linewidth=2, color='red', label='Visual Classifier') # FIXME: Change label
    ax.fill(angles, model1_accuracy, alpha=0.35, color='red')
    ax.plot(angles, model2_accuracy, '-', linewidth=2, color='blue', label='TemporalNet')      # FIXME: Change label
    ax.fill(angles, model2_accuracy, alpha=0.35, color='blue')

    # Configure small details 
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(simplified_names[:-1])
    ax.tick_params(axis='x', pad=10)  # Add padding so the ticks go more outwards
    ax.set_ylim(0, 1)
    ax.yaxis.set_ticks(np.arange(0, 1.05, 0.10))
    ax.grid(True, color="white", alpha=0.20) # FIXME
    ax.set_facecolor('#eaeaf3') # FIXME
    ax.spines['polar'].set_visible(False)
    plt.title(title, size=15, color="black") # FIXME
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1), facecolor="white") # FIXME

    # Save plot as figure
    plt.tight_layout()
    plt.savefig(f"{image_name}.png", bbox_inches="tight", facecolor="white") # FIXME
    
    # Return name mapping for reference
    simplified_to_full = {
        simplified_names[i]: experiment_names[i]
        for i in range(len(experiment_names))
    }
    return simplified_to_full



# Generate radial plot comparing YOLOv11-cls & TNet
name_map = generate_radial_plot(
    "yolov11_testing_metrics.csv", 
    "new4_tnet_metrics.csv",
    metric_col="overall_accuracy", 
    title="", # optional: add a title to radial plot
    image_name="TNet_vs_VisualClassifier"
)

# Print out mapping of simplified experiment to full experiment name
with open('Visual_TNet_Mapping.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Simplified Name', 'Full Name'])  # Header
    for key, value in name_map.items():
        writer.writerow([key, value])



"""
# Generat radial plot comparing YOLOv11 Obj Detection & AnimalTrackNet
name_map = generate_radial_plot(
    "yolov11s_exp.csv", 
    "new_combined_testing_metrics.csv",
    metric_col="overall_accuracy", 
    title="",
    image_name="AnimalTrackNet_vs_YOLOv11"
)

# Print out mapping of simplified experiment to full experiment name
with open('YOLOv11_ATN_Mapping.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Simplified Name', 'Full Name'])  # Header
    for key, value in name_map.items():
        writer.writerow([key, value])
"""