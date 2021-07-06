import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from utils import Stats


# Loading saved stats on classifier accuracies
stats = Stats(load=True, path='summaries', filename='classifier_accuracies')

# Generating the DataFrame out of the stats
stats_dict = {'stat_name':[], 'classifier':[], 'accuracy':[]}
for stat_name, stat in stats.stats.items():
    for classifier, accuracy in stat.items():
        stats_dict['stat_name'].append(stat_name)
        stats_dict['classifier'].append(classifier)
        stats_dict['accuracy'].append(accuracy)
        
stats_df = pd.DataFrame(stats_dict)

'''
stats_df['stat_name'] = stats_df['stat_name'].map({'control_vae_both':'VAE', 'control_ROI_vae_both':'VAE ROI-Reduced', 'both':'Original Features', 'both_ROI':'Original Features ROI-Reduced'})
'''

# Plotting the stats
ax = sns.barplot(x="classifier", y="accuracy", hue="stat_name", data=stats_df)

plt.show()

