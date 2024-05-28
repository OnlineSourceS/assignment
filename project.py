# https://www.kaggle.com/datasets/insiyeah/musicfeatures

import pandas as pd
import numpy as np

df = pd.read_csv('data.csv') 
print("\nHead of the DataFrame:")
print(df.head())
print("\nBasic Statistics:")
print(df.describe())
# Convert selected columns to a NumPy array
array = df[['tempo', 'beats', 'chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate']].to_numpy()

# Calculate the mean of each column
mean_values = np.mean(array, axis=0)
print("\nMean of each column:")
print(mean_values)




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Scatter plot of tempo vs. beats
plt.figure(figsize=(10, 6))
plt.scatter(df['tempo'], df['beats'], color='blue', label='Data Points')

# Calculate the regression line
x = df['tempo'].values
y = df['beats'].values
slope, intercept = np.polyfit(x, y, 1)
regression_line = slope * x + intercept

# Plot the regression line
plt.plot(x, regression_line, color='red', label='Regression Line')

# Add labels and title
plt.title('Tempo vs. Beats with Regression Line')
plt.xlabel('Tempo')
plt.ylabel('Beats')
plt.legend()
plt.show()
