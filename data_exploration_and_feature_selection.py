# data_exploration_and_feature_selection.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2

# Load the dataset
data = pd.read_csv('data/MATE1_descriptors.csv')

# Initial data exploration
print(data.info())
print(data.describe())

# Create binary target variable
data['is_inhibited'] = (data['% inhibition'] >= 50).astype(int)

# Drop non-numeric columns for correlation calculation
numeric_data = data.select_dtypes(include=[float, int])

# Correlation analysis
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()


