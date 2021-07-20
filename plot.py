import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Reading Data from CSV and removing unwanted index Column
df = pd.read_csv('data/Train.csv')
df = df.drop(columns=['Unnamed: 0'],)

label_array = np.array(df['Label'])
(emojis, counts) = np.unique(label_array, return_counts=True)

plt.style.use('dark_background')
sns.barplot(emojis, counts)
plt.xlabel(xlabel='Emoji Labels')
plt.ylabel(ylabel='Emoji Count')
plt.title('Frequency of Each Emoji Data')
plt.show()
