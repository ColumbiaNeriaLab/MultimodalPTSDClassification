from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from datautils import *

from vae import load_latents
from utils import Stats

X_list = []
Y_list = []
stat_names = []

types = ['RS', 'T1', 'DTI']

# Creating/loading stats for the classifier accuracies
stats = Stats(load=False, path='summaries', filename='classifier_accuracies')

for dataset_type in types:
    '''
    # Use this part in to train on the control_ptsd dataset
    dataset= generate_datasets(dataset_type=dataset_type, patient_type='control_ptsd', scale_features='robust')
    trans = transforms.Compose([GaussianNoise(0, 0.1, dataset.brain_columns)])
    dataset.transform = trans

    X = dataset.get_X().to_numpy()
    Y = dataset.get_Y().to_numpy().ravel()

    stat_name = 'original_{}'.format(dataset_type)

    X_list.append(X)
    Y_list.append(Y)
    stat_names.append(stat_name)
    '''
    
    # Use this part instead to train on the latent variables generated by the VAE trained on
    # the ROI-reduced control dataset applied to the control_ptsd dataset
    # To instead train on the latent variables generated by he VAE trained on the
    # original control dataset, replace net_name with 'control'
    net_name = 'control_{}'.format(dataset_type)
    fname_tag = 'both'
    stat_name = 'vae_{}_{}'.format(net_name, fname_tag)

    latent_df = load_latents(net_name, fname_tag=fname_tag)

    info_cols = ['Site', 'SubjectID']
    label_col = ['Diagnosis']
    data_cols = latent_df.columns[~latent_df.columns.isin(info_cols+label_col)]

    X = latent_df[data_cols].to_numpy()
    Y = latent_df[label_col].to_numpy().ravel()

    X_list.append(X)
    Y_list.append(Y)
    stat_names.append(stat_name)

# Uncomment this to perform TSNE plotting 
'''
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X,Y)

df_subset = pd.DataFrame({'tsne-2d-one' : tsne_results[:,0], 'tsne-2d-two' : tsne_results[:,1], 'y':Y})

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=df_subset,
    legend="full",
    alpha=0.3
)

plt.show()
'''

# Performing 10-folds cross validation on random forest, SVM, logistic regression,
# decision tree classifier, KNN, and gaussian NB

kf = KFold(n_splits=10, shuffle=True)

##rforest = RandomForestClassifier(n_estimators=100, max_depth=35)
rforest = RandomForestClassifier(n_estimators=100)
svc = SVC(class_weight='balanced', gamma='scale')
logreg = LogisticRegression(max_iter=300)
dtree = DecisionTreeClassifier()
knn = KNeighborsClassifier()
gnb = GaussianNB()

clfs = [rforest, svc, logreg, dtree, knn, gnb]
##clfs = [dtree, knn, gnb]

clf_dict = {clf.__class__.__name__ : clf for clf in clfs}

for X, Y, stat_name in zip(X_list, Y_list, stat_names):
    print("On stat {}".format(stat_name))
    for clf_name, clf in clf_dict.items():
        print("Running {}".format(clf_name))
        
        accuracies = []
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            
            clf.fit(X_train, Y_train)

            Y_pred = clf.predict(X_test)

            accuracy = np.mean(np.equal(Y_test, Y_pred))
            
            print("Accuracy =", accuracy)
            accuracies.append(accuracy)

            conf_mat = confusion_matrix(Y_test, Y_pred)
            print(conf_mat)
        
        mean_accuracy = np.mean(np.array(accuracies))
        
        # Tracking accuracy for each classifier in stats
        stats.track_stat(stat_name, clf_name, mean_accuracy)
        
        print("Mean accuracy =", mean_accuracy)
        
        stats.save()