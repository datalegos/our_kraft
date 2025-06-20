import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Make output_dir configurable via environment variable
output_dir = os.getenv("VISUALS_OUTPUT_DIR", "/mnt/data/visuals")
os.makedirs(output_dir, exist_ok=True)

# Simulate sample data
np.random.seed(42)
num_samples = 150
df = pd.DataFrame({
    'duration': np.random.exponential(scale=3.0, size=num_samples),
    'src_bytes': np.random.randint(0, 1500, size=num_samples),
    'dst_bytes': np.random.randint(0, 3000, size=num_samples),
    'wrong_fragment': np.random.randint(0, 3, size=num_samples),
    'urgent': np.random.randint(0, 2, size=num_samples),
    'Predicted_Label': np.random.choice(['Black Hole', 'Flooding', 'Grayhole', 'Scheduling', 'Normal'], size=num_samples)
})

# 1. Enhanced Scatter Plot (src vs dst bytes)
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df,
    x='src_bytes',
    y='dst_bytes',
    hue='Predicted_Label',
    palette='tab10',
    style='Predicted_Label',
    s=100,
    edgecolor='black'
)
plt.title('Enhanced Scatter Plot: Source vs Destination Bytes by Attack Type', fontsize=14)
plt.xlabel('Source Bytes')
plt.ylabel('Destination Bytes')
plt.grid(True)
plt.tight_layout()
scatter_path = os.path.join(output_dir, 'enhanced_scatter_src_dst.png')
plt.savefig(scatter_path)
plt.close()

# 2. Violin Plot (duration by attack type)
plt.figure(figsize=(10, 7))
sns.violinplot(data=df, x='Predicted_Label', y='duration', palette='Set3', inner='quartile')
plt.title('Violin Plot: Duration by Predicted Attack Type', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
violin_path = os.path.join(output_dir, 'violinplot_duration.png')
plt.savefig(violin_path)
plt.close()

# 3. Styled Correlation Heatmap
plt.figure(figsize=(8, 6))
corr = df.drop(columns=['Predicted_Label']).corr()
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    square=True,
    cbar_kws={"shrink": .8},
    linewidths=0.5
)
plt.title('Styled Correlation Heatmap', fontsize=14)
plt.tight_layout()
heatmap_path = os.path.join(output_dir, 'styled_correlation_heatmap.png')
plt.savefig(heatmap_path)
plt.close()

# 4. Donut Chart of Attack Distribution
attack_counts = df['Predicted_Label'].value_counts()
plt.figure(figsize=(7, 7))
wedges, texts, autotexts = plt.pie(
    attack_counts,
    labels=attack_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=sns.color_palette("pastel"),
    textprops={'fontsize': 12}
)
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('Donut Chart: Attack Type Distribution', fontsize=14)
plt.tight_layout()
donut_path = os.path.join(output_dir, 'donut_attack_distribution.png')
plt.savefig(donut_path)
plt.close()
