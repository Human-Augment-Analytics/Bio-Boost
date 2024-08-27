mexp2_1=1-(male2_1.groupby(['experiment_number'])['err'].sum()/(male2_1.groupby(['experiment_number'])['experiment_number'].count()))
mexp2_2=(male2_2.groupby(['experiment_number'])['acc'].sum()/(male2_2.groupby(['experiment_number'])['experiment_number'].count()))
mexp2_3=(male2_3.groupby(['experiment_number'])['acc'].sum()/(male2_3.groupby(['experiment_number'])['experiment_number'].count()))
fexp2_1=1-(female2_1.groupby(['experiment_number'])['err'].sum()/(female2_1.groupby(['experiment_number'])['experiment_number'].count()))
fexp2_2=(female2_2.groupby(['experiment_number'])['acc'].sum()/(female2_2.groupby(['experiment_number'])['experiment_number'].count()))
fexp2_3=(female2_3.groupby(['experiment_number'])['acc'].sum()/(female2_3.groupby(['experiment_number'])['experiment_number'].count()))
r2_1=pd.read_csv('/home/bshi/Documents/results/Result_2_1.csv')
r2_1['err']=abs(r2_1.label-r2_1.class_id)

unique_trials = r2_1['trial'].unique()
experiment_dict = {trial: f"exp{i+1}" for i, trial in enumerate(unique_trials)}

# Create the new column
r2_1['experiment_number'] = r2_1['trial'].map(experiment_dict)

r2_2=pd.read_csv('/home/bshi/Documents/results/Results2_2.csv')
r2_3=pd.read_csv('/home/bshi/Documents/results/Results2_3.csv')

r2_2['experiment_number'] = r2_2['trial'].map(experiment_dict)
r2_3['experiment_number'] = r2_3['trial'].map(experiment_dict)
male2_1=r2_1[r2_1.label==0]
female2_1=r2_1[r2_1.label==1]
male2_2=r2_2[r2_2.label=='male']
female2_2=r2_2[r2_2.label=='female']
male2_3=r2_3[r2_3.label=='male']
female2_3=r2_3[r2_3.label=='female']
# Data
mexp2_1=1-(male2_1.groupby(['experiment_number'])['err'].sum()/(male2_1.groupby(['experiment_number'])['experiment_number'].count()))
mexp2_2=(male2_2.groupby(['experiment_number'])['acc'].sum()/(male2_2.groupby(['experiment_number'])['experiment_number'].count()))
mexp2_3=(male2_3.groupby(['experiment_number'])['acc'].sum()/(male2_3.groupby(['experiment_number'])['experiment_number'].count()))
fexp2_1=1-(female2_1.groupby(['experiment_number'])['err'].sum()/(female2_1.groupby(['experiment_number'])['experiment_number'].count()))
fexp2_2=(female2_2.groupby(['experiment_number'])['acc'].sum()/(female2_2.groupby(['experiment_number'])['experiment_number'].count()))
fexp2_3=(female2_3.groupby(['experiment_number'])['acc'].sum()/(female2_3.groupby(['experiment_number'])['experiment_number'].count()))
categories = list(set(mexp2_1.index) | set(mexp2_2.index) | set(mexp2_3.index))
mexp2_1 = mexp2_1.reindex(categories, fill_value=0)
mexp2_2 = mexp2_2.reindex(categories, fill_value=0)
mexp2_3 = mexp2_3.reindex(categories, fill_value=0)

# Get values
values1 = mexp2_1.values
values2 = mexp2_2.values
values3 = mexp2_3.values

# Number of variables
N = len(categories)

# Create angles for each category
angles = [n / float(N) * 2 * np.pi for n in range(N)]

# Extend values and angles to close the polygon
values1 = np.concatenate((values1, [values1[0]]))
values2 = np.concatenate((values2, [values2[0]]))
values3 = np.concatenate((values3, [values3[0]]))
angles += angles[:1]

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))

# Draw the chart
ax.plot(angles, values1,  label='Manual')
ax.fill(angles, values1, alpha=0.1)
ax.plot(angles, values2,  label='Semi-Automatic')
ax.fill(angles, values2, alpha=0.1)
ax.plot(angles, values3,   label='Automatic')
ax.fill(angles, values3, alpha=0.1)

# Set the labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Add title
plt.title("Comparison of male accuracy per experiment and model type on semi-automatic validation", pad=20)
plt.savefig('./figures/fig3_title.png', dpi=300, bbox_inches='tight')
# Adjust layout and display
plt.tight_layout()
plt.show()
r2_1=pd.read_csv('/home/bshi/Documents/results/Result_2_1.csv')
r2_1['err']=abs(r2_1.label-r2_1.class_id)

unique_trials = r2_1['trial'].unique()
experiment_dict = {trial: f"exp{i+1}" for i, trial in enumerate(unique_trials)}

# Create the new column
r2_1['experiment_number'] = r2_1['trial'].map(experiment_dict)

r2_2=pd.read_csv('/home/bshi/Documents/results/Results2_2.csv')
r2_3=pd.read_csv('/home/bshi/Documents/results/Results2_3.csv')

r2_2['experiment_number'] = r2_2['trial'].map(experiment_dict)
r2_3['experiment_number'] = r2_3['trial'].map(experiment_dict)
male2_1=r2_1[r2_1.label==0]
female2_1=r2_1[r2_1.label==1]
male2_2=r2_2[r2_2.label=='male']
female2_2=r2_2[r2_2.label=='female']
male2_3=r2_3[r2_3.label=='male']
female2_3=r2_3[r2_3.label=='female']
# Data
mexp2_1=1-(male2_1.groupby(['experiment_number'])['err'].sum()/(male2_1.groupby(['experiment_number'])['experiment_number'].count()))
mexp2_2=(male2_2.groupby(['experiment_number'])['acc'].sum()/(male2_2.groupby(['experiment_number'])['experiment_number'].count()))
mexp2_3=(male2_3.groupby(['experiment_number'])['acc'].sum()/(male2_3.groupby(['experiment_number'])['experiment_number'].count()))
fexp2_1=1-(female2_1.groupby(['experiment_number'])['err'].sum()/(female2_1.groupby(['experiment_number'])['experiment_number'].count()))
fexp2_2=(female2_2.groupby(['experiment_number'])['acc'].sum()/(female2_2.groupby(['experiment_number'])['experiment_number'].count()))
fexp2_3=(female2_3.groupby(['experiment_number'])['acc'].sum()/(female2_3.groupby(['experiment_number'])['experiment_number'].count()))

categories = list(set(mexp2_1.index) | set(mexp2_2.index) | set(mexp2_3.index))
mexp2_1 = mexp2_1.reindex(categories, fill_value=0)
mexp2_2 = mexp2_2.reindex(categories, fill_value=0)
mexp2_3 = mexp2_3.reindex(categories, fill_value=0)

# Get values
values1 = mexp2_1.values
values2 = mexp2_2.values
values3 = mexp2_3.values

# Number of variables
N = len(categories)

# Create angles for each category
angles = [n / float(N) * 2 * np.pi for n in range(N)]

# Extend values and angles to close the polygon
values1 = np.concatenate((values1, [values1[0]]))
values2 = np.concatenate((values2, [values2[0]]))
values3 = np.concatenate((values3, [values3[0]]))
angles += angles[:1]

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))

# Draw the chart
ax.plot(angles, values1,  label='Manual')
ax.fill(angles, values1, alpha=0.1)
ax.plot(angles, values2,  label='Semi-Automatic')
ax.fill(angles, values2, alpha=0.1)
ax.plot(angles, values3,   label='Automatic')
ax.fill(angles, values3, alpha=0.1)

# Set the labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Add title
plt.title("Comparison of male accuracy per experiment and model type on semi-automatic validation", pad=20)
plt.savefig('./figures/fig3_title.png', dpi=300, bbox_inches='tight')
# Adjust layout and display
plt.tight_layout()
plt.show()
categories = list(set(mexp2_1.index) | set(mexp2_2.index) | set(mexp2_3.index))
mexp2_1 = mexp2_1.reindex(categories, fill_value=0)
mexp2_2 = mexp2_2.reindex(categories, fill_value=0)
mexp2_3 = mexp2_3.reindex(categories, fill_value=0)

# Get values
values1 = mexp2_1.values
values2 = mexp2_2.values
values3 = mexp2_3.values

# Number of variables
N = len(categories)

# Create angles for each category
angles = [n / float(N) * 2 * np.pi for n in range(N)]

# Extend values and angles to close the polygon
values1 = np.concatenate((values1, [values1[0]]))
values2 = np.concatenate((values2, [values2[0]]))
values3 = np.concatenate((values3, [values3[0]]))
angles += angles[:1]

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))

# Draw the chart
ax.plot(angles, values1, )
ax.fill(angles, values1, alpha=0.1)
ax.plot(angles, values2, )
ax.fill(angles, values2, alpha=0.1)
ax.plot(angles, values3,   )
ax.fill(angles, values3, alpha=0.1)

# Set the labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Add title
#plt.title("Comparison of male accuracy per experiment and model type on semi-automatic validation", pad=20)
plt.savefig('./figures/fig3.png', dpi=300, bbox_inches='tight')
# Adjust layout and display
plt.tight_layout()
plt.show()
categories = list(set(fexp2_1.index) | set(fexp2_2.index) | set(fexp2_3.index))
fexp2_1 = fexp2_1.reindex(categories, fill_value=0)
fexp2_2 = fexp2_2.reindex(categories, fill_value=0)
fexp2_3 = fexp2_3.reindex(categories, fill_value=0)

# Get values
values1 = fexp2_1.values
values2 = fexp2_2.values
values3 = fexp2_3.values

# Number of variables
N = len(categories)

# Create angles for each category
angles = [n / float(N) * 2 * np.pi for n in range(N)]

# Extend values and angles to close the polygon
values1 = np.concatenate((values1, [values1[0]]))
values2 = np.concatenate((values2, [values2[0]]))
values3 = np.concatenate((values3, [values3[0]]))
angles += angles[:1]

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))

# Draw the chart
ax.plot(angles, values1,  label='Manual')
ax.fill(angles, values1, alpha=0.1)
ax.plot(angles, values2,  label='Semi-Automatic')
ax.fill(angles, values2, alpha=0.1)
ax.plot(angles, values3,   label='Automatic')
ax.fill(angles, values3, alpha=0.1)

# Set the labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Add title
plt.title("Comparison of female accuracy per experiment and model type on semi-automatic validation", pad=20)
plt.savefig('./figures/fig4_title.png', dpi=300, bbox_inches='tight')
# Adjust layout and display
plt.tight_layout()
plt.show(
categories = list(set(fexp2_1.index) | set(fexp2_2.index) | set(fexp2_3.index))
fexp2_1 = fexp2_1.reindex(categories, fill_value=0)
fexp2_2 = fexp2_2.reindex(categories, fill_value=0)
fexp2_3 = fexp2_3.reindex(categories, fill_value=0)

# Get values
values1 = fexp2_1.values
values2 = fexp2_2.values
values3 = fexp2_3.values

# Number of variables
N = len(categories)

# Create angles for each category
angles = [n / float(N) * 2 * np.pi for n in range(N)]

# Extend values and angles to close the polygon
values1 = np.concatenate((values1, [values1[0]]))
values2 = np.concatenate((values2, [values2[0]]))
values3 = np.concatenate((values3, [values3[0]]))
angles += angles[:1]

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))

# Draw the chart
ax.plot(angles, values1,  label='Manual')
ax.fill(angles, values1, alpha=0.1)
ax.plot(angles, values2,  label='Semi-Automatic')
ax.fill(angles, values2, alpha=0.1)
ax.plot(angles, values3,   label='Automatic')
ax.fill(angles, values3, alpha=0.1)

# Set the labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Add title
plt.title("Comparison of female accuracy per experiment and model type on semi-automatic validation", pad=20)
plt.savefig('./figures/fig4_title.png', dpi=300, bbox_inches='tight')
# Adjust layout and display
plt.tight_layout()
plt.show()
categories = list(set(fexp2_1.index) | set(fexp2_2.index) | set(fexp2_3.index))
fexp2_1 = fexp2_1.reindex(categories, fill_value=0)
fexp2_2 = fexp2_2.reindex(categories, fill_value=0)
fexp2_3 = fexp2_3.reindex(categories, fill_value=0)

# Get values
values1 = fexp2_1.values
values2 = fexp2_2.values
values3 = fexp2_3.values

# Number of variables
N = len(categories)

# Create angles for each category
angles = [n / float(N) * 2 * np.pi for n in range(N)]

# Extend values and angles to close the polygon
values1 = np.concatenate((values1, [values1[0]]))
values2 = np.concatenate((values2, [values2[0]]))
values3 = np.concatenate((values3, [values3[0]]))
angles += angles[:1]

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))

# Draw the chart
ax.plot(angles, values1, )
ax.fill(angles, values1, alpha=0.1)
ax.plot(angles, values2, )
ax.fill(angles, values2, alpha=0.1)
ax.plot(angles, values3,  )
ax.fill(angles, values3, alpha=0.1)

# Set the labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Add title
#plt.title("Comparison of female accuracy per experiment and model type on semi-automatic validation", pad=20)
plt.savefig('./figures/fig4.png', dpi=300, bbox_inches='tight')
# Adjust layout and display
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Combine all indices
all_indices = set(mexp2_1.index) | set(mexp2_2.index) | set(mexp2_3.index) | \
              set(fexp2_1.index) | set(fexp2_2.index) | set(fexp2_3.index)

# Reindex all series to have the same index
for series in [mexp2_1, mexp2_2, mexp2_3, fexp2_1, fexp2_2, fexp2_3]:
    series = series.reindex(all_indices, fill_value=0)

# Create a DataFrame with all the data
df = pd.DataFrame({
    'Manual (Male)': mexp2_1,
    'Manual (Female)': fexp2_1,
    'Semi-Auto (Male)': mexp2_2,
    'Semi-Auto (Female)': fexp2_2,
    'Auto (Male)': mexp2_3,
    'Auto (Female)': fexp2_3
})

# Calculate differences
df['Manual (Diff)'] = df['Manual (Male)'] - df['Manual (Female)']
df['Semi-Auto (Diff)'] = df['Semi-Auto (Male)'] - df['Semi-Auto (Female)']
df['Auto (Diff)'] = df['Auto (Male)'] - df['Auto (Female)']

# Prepare data for plotting
categories = df.index
x = np.arange(len(categories))
width = 0.25

# Create the plot
fig, ax = plt.subplots(figsize=(15, 8))

# Plot bars
rects1 = ax.bar(x - width, df['Manual (Diff)'], width, label='Manual', color='skyblue')
rects2 = ax.bar(x, df['Semi-Auto (Diff)'], width, label='Semi-Automatic', color='orange')
rects3 = ax.bar(x + width, df['Auto (Diff)'], width, label='Automatic', color='green')

# Customize the plot
ax.set_ylabel('Accuracy Difference (Male - Female)')
ax.set_title('Gender Accuracy Difference by Experiment and Model Type (Semi-automatic Validation)')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.legend()

# Add a horizontal line at y=0
ax.axhline(y=0, color='r', linestyle='-', linewidth=0.5)

# Add value labels on the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()

# Save the figure
#plt.savefig('./figures/gender_accuracy_difference_semiauto.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
import matplotlib.pyplot as plt
import pandas as pd

# Combine all indices
all_indices = set(mexp2_1.index) | set(mexp2_2.index) | set(mexp2_3.index) | \
              set(fexp2_1.index) | set(fexp2_2.index) | set(fexp2_3.index)

# Reindex all series to have the same index
for series in [mexp2_1, mexp2_2, mexp2_3, fexp2_1, fexp2_2, fexp2_3]:
    series = series.reindex(all_indices, fill_value=0)

# Create a DataFrame with all the data
df = pd.DataFrame({
    'Manual (Male)': mexp2_1,
    'Manual (Female)': fexp2_1,
    'Semi-Auto (Male)': mexp2_2,
    'Semi-Auto (Female)': fexp2_2,
    'Auto (Male)': mexp2_3,
    'Auto (Female)': fexp2_3
})

# Calculate differences
df['Manual (Diff)'] = df['Manual (Male)'] - df['Manual (Female)']
df['Semi-Auto (Diff)'] = df['Semi-Auto (Male)'] - df['Semi-Auto (Female)']
df['Auto (Diff)'] = df['Auto (Male)'] - df['Auto (Female)']

# Prepare data for plotting
diff_data = [df['Manual (Diff)'], df['Semi-Auto (Diff)'], df['Auto (Diff)']]
labels = ['Manual', 'Semi-Automatic', 'Automatic']

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Create box plot
bp = ax.boxplot(diff_data, labels=labels, patch_artist=True)

# Customize colors
colors = ['skyblue', 'orange', 'green']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Customize the plot
ax.set_ylabel('Accuracy Difference (Male - Female)')
ax.set_title('Distribution of Gender Accuracy Difference by Model Type (Semi-automatic Validation)')

# Add a horizontal line at y=0
ax.axhline(y=0, color='r', linestyle='--', linewidth=0.5)

# Add grid lines
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

plt.tight_layout()

# Save the figure
#plt.savefig('./figures/gender_accuracy_difference_boxplot_semiauto.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Combine all indices
all_indices = set(mexp2_1.index) | set(mexp2_2.index) | set(mexp2_3.index) | \
              set(fexp2_1.index) | set(fexp2_2.index) | set(fexp2_3.index)

# Reindex all series to have the same index
for series in [mexp2_1, mexp2_2, mexp2_3, fexp2_1, fexp2_2, fexp2_3]:
    series = series.reindex(all_indices, fill_value=0)

# Create a DataFrame with all the data
df = pd.DataFrame({
    'Manual (Male)': mexp2_1,
    'Manual (Female)': fexp2_1,
    'Semi-Auto (Male)': mexp2_2,
    'Semi-Auto (Female)': fexp2_2,
    'Auto (Male)': mexp2_3,
    'Auto (Female)': fexp2_3
})

# Prepare data for plotting
data = [df['Manual (Male)'], df['Manual (Female)'],
        df['Semi-Auto (Male)'], df['Semi-Auto (Female)'],
        df['Auto (Male)'], df['Auto (Female)']]

# Create the plot
fig, ax = plt.subplots(figsize=(15, 8))

# Create box plot
bp = ax.boxplot(data, patch_artist=True)

# Customize colors
colors = ['skyblue', 'pink', 'lightgreen', 'lightcoral', 'lightskyblue', 'lightpink']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Customize the plot
ax.set_ylabel('Accuracy')
ax.set_title('Distribution of Accuracy by Gender and Model Type (Semi-automatic Validation)')

# Set x-tick labels
ax.set_xticklabels(['Manual\nMale', 'Manual\nFemale', 
                    'Semi-Auto\nMale', 'Semi-Auto\nFemale', 
                    'Auto\nMale', 'Auto\nFemale'])

# Add grid lines
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

# Add legend
legend_elements = [plt.Rectangle((0,0),1,1, facecolor='skyblue', edgecolor='black', label='Male'),
                   plt.Rectangle((0,0),1,1, facecolor='pink', edgecolor='black', label='Female')]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()

# Save the figure
plt.savefig('./figures/gender_accuracy_boxplot_semiauto.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
2_1
r2_1
r2_1.columns
r2_1.a2_track_id
r2_1.label
r2_2[r2_2.track_id==428].label.unqiue()
r2_2[r2_2.track_id==428].label
fexp2_1
mexp2_1
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Ensure both series have the same index
all_indices = set(mexp2_2.index) | set(fexp2_2.index)
mexp2_2 = mexp2_2.reindex(all_indices, fill_value=np.nan)
fexp2_2 = fexp2_2.reindex(all_indices, fill_value=np.nan)

# Create a DataFrame with the data
df = pd.DataFrame({
    'Male': mexp2_2,
    'Female': fexp2_2
})

# Remove any rows with NaN values
df = df.dropna()

# Sort the DataFrame by the experiment number
df = df.sort_index()

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Create box plot
bp = ax.boxplot([df['Male'], df['Female']], positions=[1, 2], patch_artist=True)

# Customize colors
colors = ['skyblue', 'pink']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Add lines connecting male and female values
for i, (_, row) in enumerate(df.iterrows()):
    ax.plot([1, 2], [row['Male'], row['Female']], 'k-', alpha=0.3)

# Customize the plot
ax.set_ylabel('Accuracy')
ax.set_title('Distribution of Accuracy by Gender for Semi-Automatic Model')
ax.set_xticks([1, 2])
ax.set_xticklabels(['Male', 'Female'])

# Add grid lines
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

# Add legend
legend_elements = [plt.Rectangle((0,0),1,1, facecolor='skyblue', edgecolor='black', label='Male'),
                   plt.Rectangle((0,0),1,1, facecolor='pink', edgecolor='black', label='Female'),
                   plt.Line2D([0], [0], color='k', alpha=0.3, label='Same Experiment')]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()

# Save the figure
#plt.savefig('./figures/semi_auto_gender_accuracy_boxplot_with_lines.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
r2_1=pd.read_csv('/home/bshi/Documents/results/Result_2_1.csv')
r2_1['err']=abs(r2_1.label-r2_1.class_id)

unique_trials = r1_1['trial'].unique()
experiment_dict = {trial: f"exp{i+1}" for i, trial in enumerate(unique_trials)}

# Create the new column
r2_1['experiment_number'] = r2_1['trial'].map(experiment_dict)

r2_2=pd.read_csv('/home/bshi/Documents/results/Results2_2.csv')
r2_3=pd.read_csv('/home/bshi/Documents/results/Results2_3.csv')

r2_2['experiment_number'] = r2_2['trial'].map(experiment_dict)
r2_3['experiment_number'] = r2_3['trial'].map(experiment_dict)
male2_1=r2_1[r2_1.label==0]
female2_1=r2_1[r2_1.label==1]
male2_2=r2_2[r2_2.label=='male']
female2_2=r2_2[r2_2.label=='female']
male2_3=r2_3[r2_3.label=='male']
female2_3=r2_3[r2_3.label=='female']
# Data
mexp2_1=1-(male2_1.groupby(['experiment_number'])['err'].sum()/(male2_1.groupby(['experiment_number'])['experiment_number'].count()))
mexp2_2=(male2_2.groupby(['experiment_number'])['acc'].sum()/(male2_2.groupby(['experiment_number'])['experiment_number'].count()))
mexp2_3=(male2_3.groupby(['experiment_number'])['acc'].sum()/(male2_3.groupby(['experiment_number'])['experiment_number'].count()))
fexp2_1=1-(female2_1.groupby(['experiment_number'])['err'].sum()/(female2_1.groupby(['experiment_number'])['experiment_number'].count()))
fexp2_2=(female2_2.groupby(['experiment_number'])['acc'].sum()/(female2_2.groupby(['experiment_number'])['experiment_number'].count()))
fexp2_3=(female2_3.groupby(['experiment_number'])['acc'].sum()/(female2_3.groupby(['experiment_number'])['experiment_number'].count()))

categories = list(set(mexp2_1.index) | set(mexp2_2.index) | set(mexp2_3.index))
mexp2_1 = mexp2_1.reindex(categories, fill_value=0)
mexp2_2 = mexp2_2.reindex(categories, fill_value=0)
mexp2_3 = mexp2_3.reindex(categories, fill_value=0)

# Get values
values1 = mexp2_1.values
values2 = mexp2_2.values
values3 = mexp2_3.values

# Number of variables
N = len(categories)

# Create angles for each category
angles = [n / float(N) * 2 * np.pi for n in range(N)]

# Extend values and angles to close the polygon
values1 = np.concatenate((values1, [values1[0]]))
values2 = np.concatenate((values2, [values2[0]]))
values3 = np.concatenate((values3, [values3[0]]))
angles += angles[:1]

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))

# Draw the chart
ax.plot(angles, values1,  label='Manual')
ax.fill(angles, values1, alpha=0.1)
ax.plot(angles, values2,  label='Semi-Automatic')
ax.fill(angles, values2, alpha=0.1)
ax.plot(angles, values3,   label='Automatic')
ax.fill(angles, values3, alpha=0.1)

# Set the labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Add title
plt.title("Comparison of male accuracy per experiment and model type on semi-automatic validation", pad=20)
plt.savefig('./figures/fig3.png', dpi=300, bbox_inches='tight')
# Adjust layout and display
plt.tight_layout()
plt.show()
r2_1=pd.read_csv('/home/bshi/Documents/results/Result_2_1.csv')
r2_1['err']=abs(r2_1.label-r2_1.class_id)

unique_trials = r1_1['trial_id'].unique()
experiment_dict = {trial: f"exp{i+1}" for i, trial in enumerate(unique_trials)}

# Create the new column
r2_1['experiment_number'] = r2_1['trial'].map(experiment_dict)

r2_2=pd.read_csv('/home/bshi/Documents/results/Results2_2.csv')
r2_3=pd.read_csv('/home/bshi/Documents/results/Results2_3.csv')

r2_2['experiment_number'] = r2_2['trial'].map(experiment_dict)
r2_3['experiment_number'] = r2_3['trial'].map(experiment_dict)
male2_1=r2_1[r2_1.label==0]
female2_1=r2_1[r2_1.label==1]
male2_2=r2_2[r2_2.label=='male']
female2_2=r2_2[r2_2.label=='female']
male2_3=r2_3[r2_3.label=='male']
female2_3=r2_3[r2_3.label=='female']
# Data
mexp2_1=1-(male2_1.groupby(['experiment_number'])['err'].sum()/(male2_1.groupby(['experiment_number'])['experiment_number'].count()))
mexp2_2=(male2_2.groupby(['experiment_number'])['acc'].sum()/(male2_2.groupby(['experiment_number'])['experiment_number'].count()))
mexp2_3=(male2_3.groupby(['experiment_number'])['acc'].sum()/(male2_3.groupby(['experiment_number'])['experiment_number'].count()))
fexp2_1=1-(female2_1.groupby(['experiment_number'])['err'].sum()/(female2_1.groupby(['experiment_number'])['experiment_number'].count()))
fexp2_2=(female2_2.groupby(['experiment_number'])['acc'].sum()/(female2_2.groupby(['experiment_number'])['experiment_number'].count()))
fexp2_3=(female2_3.groupby(['experiment_number'])['acc'].sum()/(female2_3.groupby(['experiment_number'])['experiment_number'].count()))

categories = list(set(mexp2_1.index) | set(mexp2_2.index) | set(mexp2_3.index))
mexp2_1 = mexp2_1.reindex(categories, fill_value=0)
mexp2_2 = mexp2_2.reindex(categories, fill_value=0)
mexp2_3 = mexp2_3.reindex(categories, fill_value=0)

# Get values
values1 = mexp2_1.values
values2 = mexp2_2.values
values3 = mexp2_3.values

# Number of variables
N = len(categories)

# Create angles for each category
angles = [n / float(N) * 2 * np.pi for n in range(N)]

# Extend values and angles to close the polygon
values1 = np.concatenate((values1, [values1[0]]))
values2 = np.concatenate((values2, [values2[0]]))
values3 = np.concatenate((values3, [values3[0]]))
angles += angles[:1]

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))

# Draw the chart
ax.plot(angles, values1,  label='Manual')
ax.fill(angles, values1, alpha=0.1)
ax.plot(angles, values2,  label='Semi-Automatic')
ax.fill(angles, values2, alpha=0.1)
ax.plot(angles, values3,   label='Automatic')
ax.fill(angles, values3, alpha=0.1)

# Set the labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Add title
plt.title("Comparison of male accuracy per experiment and model type on semi-automatic validation", pad=20)
plt.savefig('./figures/fig3.png', dpi=300, bbox_inches='tight')
# Adjust layout and display
plt.tight_layout()
plt.show()

categories = list(set(fexp2_1.index) | set(fexp2_2.index) | set(fexp2_3.index))
fexp2_1 = fexp2_1.reindex(categories, fill_value=0)
fexp2_2 = fexp2_2.reindex(categories, fill_value=0)
fexp2_3 = fexp2_3.reindex(categories, fill_value=0)

# Get values
values1 = fexp2_1.values
values2 = fexp2_2.values
values3 = fexp2_3.values

# Number of variables
N = len(categories)

# Create angles for each category
angles = [n / float(N) * 2 * np.pi for n in range(N)]

# Extend values and angles to close the polygon
values1 = np.concatenate((values1, [values1[0]]))
values2 = np.concatenate((values2, [values2[0]]))
values3 = np.concatenate((values3, [values3[0]]))
angles += angles[:1]

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))

# Draw the chart
ax.plot(angles, values1, )
ax.fill(angles, values1, alpha=0.1)
ax.plot(angles, values2, )
ax.fill(angles, values2, alpha=0.1)
ax.plot(angles, values3,  )
ax.fill(angles, values3, alpha=0.1)

# Set the labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Add title
#plt.title("Comparison of female accuracy per experiment and model type on semi-automatic validation", pad=20)
plt.savefig('./figures/fig4.png', dpi=300, bbox_inches='tight')
# Adjust layout and display
plt.tight_layout()
plt.show()
categories = list(set(mexp2_1.index) | set(mexp2_2.index) | set(mexp2_3.index))
mexp2_1 = mexp2_1.reindex(categories, fill_value=0)
mexp2_2 = mexp2_2.reindex(categories, fill_value=0)
mexp2_3 = mexp2_3.reindex(categories, fill_value=0)

# Get values
values1 = mexp2_1.values
values2 = mexp2_2.values
values3 = mexp2_3.values

# Number of variables
N = len(categories)

# Create angles for each category
angles = [n / float(N) * 2 * np.pi for n in range(N)]

# Extend values and angles to close the polygon
values1 = np.concatenate((values1, [values1[0]]))
values2 = np.concatenate((values2, [values2[0]]))
values3 = np.concatenate((values3, [values3[0]]))
angles += angles[:1]

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))

# Draw the chart
ax.plot(angles, values1,  )
ax.fill(angles, values1, alpha=0.1)
ax.plot(angles, values2, )
ax.fill(angles, values2, alpha=0.1)
ax.plot(angles, values3,   )
ax.fill(angles, values3, alpha=0.1)

# Set the labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Add title
#plt.title("Comparison of male accuracy per experiment and model type on semi-automatic validation", pad=20)
plt.savefig('./figures/fig3.png', dpi=300, bbox_inches='tight')
# Adjust layout and display
plt.tight_layout()
plt.show()
r2_1=pd.read_csv('/home/bshi/Documents/results/Result_2_1.csv')
r2_1['err']=abs(r2_1.label-r2_1.class_id)

unique_trials = r1_1['trial_id'].unique()
experiment_dict = {trial: f"exp{i+1}" for i, trial in enumerate(unique_trials)}

# Create the new column
r2_1['experiment_number'] = r2_1['trial'].map(experiment_dict)

r2_2=pd.read_csv('/home/bshi/Documents/results/Results2_2.csv')
r2_3=pd.read_csv('/home/bshi/Documents/results/Results2_3.csv')

r2_2['experiment_number'] = r2_2['trial'].map(experiment_dict)
r2_3['experiment_number'] = r2_3['trial'].map(experiment_dict)
male2_1=r2_1[r2_1.label==0]
female2_1=r2_1[r2_1.label==1]
male2_2=r2_2[r2_2.label=='male']
female2_2=r2_2[r2_2.label=='female']
male2_3=r2_3[r2_3.label=='male']
female2_3=r2_3[r2_3.label=='female']
# Data
mexp2_1=1-(male2_1.groupby(['experiment_number'])['err'].sum()/(male2_1.groupby(['experiment_number'])['experiment_number'].count()))
mexp2_2=(male2_2.groupby(['experiment_number'])['acc'].sum()/(male2_2.groupby(['experiment_number'])['experiment_number'].count()))
mexp2_3=(male2_3.groupby(['experiment_number'])['acc'].sum()/(male2_3.groupby(['experiment_number'])['experiment_number'].count()))
fexp2_1=1-(female2_1.groupby(['experiment_number'])['err'].sum()/(female2_1.groupby(['experiment_number'])['experiment_number'].count()))
fexp2_2=(female2_2.groupby(['experiment_number'])['acc'].sum()/(female2_2.groupby(['experiment_number'])['experiment_number'].count()))
fexp2_3=(female2_3.groupby(['experiment_number'])['acc'].sum()/(female2_3.groupby(['experiment_number'])['experiment_number'].count()))

categories = list(set(mexp2_1.index) | set(mexp2_2.index) | set(mexp2_3.index))
mexp2_1 = mexp2_1.reindex(categories, fill_value=0)
mexp2_2 = mexp2_2.reindex(categories, fill_value=0)
mexp2_3 = mexp2_3.reindex(categories, fill_value=0)

# Get values
values1 = mexp2_1.values
values2 = mexp2_2.values
values3 = mexp2_3.values

# Number of variables
N = len(categories)

# Create angles for each category
angles = [n / float(N) * 2 * np.pi for n in range(N)]

# Extend values and angles to close the polygon
values1 = np.concatenate((values1, [values1[0]]))
values2 = np.concatenate((values2, [values2[0]]))
values3 = np.concatenate((values3, [values3[0]]))
angles += angles[:1]

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))

# Draw the chart
ax.plot(angles, values1,  label='Manual')
ax.fill(angles, values1, alpha=0.1)
ax.plot(angles, values2,  label='Semi-Automatic')
ax.fill(angles, values2, alpha=0.1)
ax.plot(angles, values3,   label='Automatic')
ax.fill(angles, values3, alpha=0.1)

# Set the labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Add title
plt.title("Comparison of male accuracy per experiment and model type on semi-automatic validation", pad=20)
plt.savefig('./figures/fig3_title.png', dpi=300, bbox_inches='tight')
# Adjust layout and display
plt.tight_layout()
plt.show()

categories = list(set(fexp2_1.index) | set(fexp2_2.index) | set(fexp2_3.index))
fexp2_1 = fexp2_1.reindex(categories, fill_value=0)
fexp2_2 = fexp2_2.reindex(categories, fill_value=0)
fexp2_3 = fexp2_3.reindex(categories, fill_value=0)

# Get values
values1 = fexp2_1.values
values2 = fexp2_2.values
values3 = fexp2_3.values

# Number of variables
N = len(categories)

# Create angles for each category
angles = [n / float(N) * 2 * np.pi for n in range(N)]

# Extend values and angles to close the polygon
values1 = np.concatenate((values1, [values1[0]]))
values2 = np.concatenate((values2, [values2[0]]))
values3 = np.concatenate((values3, [values3[0]]))
angles += angles[:1]

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))

# Draw the chart
ax.plot(angles, values1,  label='Manual')
ax.fill(angles, values1, alpha=0.1)
ax.plot(angles, values2,  label='Semi-Automatic')
ax.fill(angles, values2, alpha=0.1)
ax.plot(angles, values3,   label='Automatic')
ax.fill(angles, values3, alpha=0.1)

# Set the labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Add title
plt.title("Comparison of female accuracy per experiment and model type on semi-automatic validation", pad=20)
plt.savefig('./figures/fig4_title.png', dpi=300, bbox_inches='tight')
# Adjust layout and display
plt.tight_layout()
plt.show()
experiment_dict
experiment_dict = {trial: f"exp{i+1}" for i, trial in enumerate(unique_trials)}
experiment_dict
trail_dif=pd.read_csv('/home/bshi/Documents/final_PBT_acc.csv')
trail_df
trail_dif
adf=pd.read_csv('/home/bshi/Documents/final_PBT_acc.csv')
adf.columns
adf['PBT_acc']
adf['PBT_acc']-adf['PBT_racc']
adf['fit']=adf['PBT_acc']-adf['PBT_racc']
adf.columns
adf[adf.fit==max(adf.fit)]
adf[adf.fit==min(adf.fit)]
adf
adf=adf[adf.trial.isin(list(experiment_dict.keys()))]
adf
adf[adf.mcgrath_score<=2]
adf[adf.mcgrath_score<=3]
adf[adf.mcgrath_score<=5]
adf[adf.mcgrath_score<=7]
adf[adf.mcgrath_score<=7].trials
adf[adf.mcgrath_score<=7].trial
poor=adf[adf.mcgrath_score<=5].trial
poor
poor=adf[adf.mcgrath_score<=5]
poor
high
low=adf[adf.mcgrath_score<=7].trial
high=adf[adf.mcgrath_score>7].trial
poor=adf[adf.mcgrath_score<=5].trial
high
low
r2_2.columns
r2_2[r2_2.experiment_number=='exp21']
test=r2_2[r2_2.experiment_number=='exp21']
test1=r2_2[r2_2.experiment_number=='exp21']
test2=r2_1[r2_1.experiment_number=='exp21']
test1[test1.label=='female'].groupby('track_id')['acc'].mean()
test2[test2.label==1].groupby('track_id')['err'].mean()
test2[test2.label==1].groupby('a2_track_id')['err'].mean()
test1[test1.track_id=='14814']
test1[test1.track_id==14814]
test1[test1.track_id==14814].base_name

## ---(Mon Jun 24 00:27:21 2024)---
adf=pd.read_csv('/home/bshi/Documents/final_PBT_acc.csv')
test1[test1.label=='female'].groupby('track_id')['acc'].mean()
test1=r2_2[r2_2.experiment_number=='exp21']
r2_1=pd.read_csv('/home/bshi/Documents/results/Result_2_1.csv')
r2_1['err']=abs(r2_1.label-r2_1.class_id)
unique_trials = r1_1['trial_id'].unique()
experiment_dict = {trial: f"exp{i+1}" for i, trial in enumerate(unique_trials)}
# Create the new column
r2_1['experiment_number'] = r2_1['trial'].map(experiment_dict)
r2_2=pd.read_csv('/home/bshi/Documents/results/Results2_2.csv')
r2_3=pd.read_csv('/home/bshi/Documents/results/Results2_3.csv')
r1_1=pd.read_csv('/home/bshi/Documents/results/Result_1_1.csv')
r2_1=pd.read_csv('/home/bshi/Documents/results/Result_2_1.csv')
r2_1['err']=abs(r2_1.label-r2_1.class_id)

unique_trials = r1_1['trial_id'].unique()
experiment_dict = {trial: f"exp{i+1}" for i, trial in enumerate(unique_trials)}

# Create the new column
r2_1['experiment_number'] = r2_1['trial'].map(experiment_dict)

r2_2=pd.read_csv('/home/bshi/Documents/results/Results2_2.csv')
r2_3=pd.read_csv('/home/bshi/Documents/results/Results2_3.csv')

r2_2['experiment_number'] = r2_2['trial'].map(experiment_dict)
r2_3['experiment_number'] = r2_3['trial'].map(experiment_dict)
adf=pd.read_csv('/home/bshi/Documents/final_PBT_acc.csv')
import pandas as pd
r1_1=pd.read_csv('/home/bshi/Documents/results/Result_1_1.csv')
r2_1=pd.read_csv('/home/bshi/Documents/results/Result_2_1.csv')
r2_1['err']=abs(r2_1.label-r2_1.class_id)

unique_trials = r1_1['trial_id'].unique()
experiment_dict = {trial: f"exp{i+1}" for i, trial in enumerate(unique_trials)}

# Create the new column
r2_1['experiment_number'] = r2_1['trial'].map(experiment_dict)

r2_2=pd.read_csv('/home/bshi/Documents/results/Results2_2.csv')
r2_3=pd.read_csv('/home/bshi/Documents/results/Results2_3.csv')

r2_2['experiment_number'] = r2_2['trial'].map(experiment_dict)
r2_3['experiment_number'] = r2_3['trial'].map(experiment_dict)
adf=pd.read_csv('/home/bshi/Documents/final_PBT_acc.csv')
import pandas as pd
r1_1=pd.read_csv('/home/bshi/Documents/results/Results1_1.csv')
r2_1=pd.read_csv('/home/bshi/Documents/results/Result_2_1.csv')
r2_1['err']=abs(r2_1.label-r2_1.class_id)

unique_trials = r1_1['trial_id'].unique()
experiment_dict = {trial: f"exp{i+1}" for i, trial in enumerate(unique_trials)}

# Create the new column
r2_1['experiment_number'] = r2_1['trial'].map(experiment_dict)

r2_2=pd.read_csv('/home/bshi/Documents/results/Results2_2.csv')
r2_3=pd.read_csv('/home/bshi/Documents/results/Results2_3.csv')

r2_2['experiment_number'] = r2_2['trial'].map(experiment_dict)
r2_3['experiment_number'] = r2_3['trial'].map(experiment_dict)
adf=pd.read_csv('/home/bshi/Documents/final_PBT_acc.csv')
r1_1.columns
