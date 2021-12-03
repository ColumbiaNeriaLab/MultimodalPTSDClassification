import pandas as pd
import numpy as np
from numpy import interp
from numpy import sqrt
from numpy import argmax
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import binarize
from sklearn.preprocessing import RobustScaler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer


def single_site_analysis(model, data,
                         drop=['Unnamed: 0', 'Diagnosis', 'Age', 'Sex', 'SubjectID', 'Site', 'CurrPTSDdx']):
    '''
    :param model: RandomForestClassifier() or SVC()
    :param data: data matrix with "Site" column
    :param drop: list of non-data columns
    :return: test_accuracy, test_auc, test_specificity, test_sensitivity
    '''

    accuracy_mean = []
    accuracy_std = []
    scoring = {'accuracy': make_scorer(accuracy_score),
               'auc': 'roc_auc',
               'specificity': make_scorer(recall_score, pos_label=0),
               'sensitivity': make_scorer(recall_score, pos_label=1)
               }

    sites = data.Site.unique()
    res = {}

    # test balanced sites
    for s in sites:
        temp = data[data.Site == s]
        X = temp.drop(columns=drop, errors='ignore')
        y = temp['Diagnosis']

        # drop empty columns and impute
        nans = X.columns[X.isnull().sum() == X.shape[0]]
        X.drop(columns=nans, inplace=True)
        X.fillna(X.mean(), inplace=True)

        steps = [('under', RandomUnderSampler()), ('scale', RobustScaler()), ('model', model)]
        pipeline = Pipeline(steps=steps)

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        res[s] = cross_validate(pipeline, X, y, scoring=scoring, cv=cv)

        accuracy_mean.append(np.mean(res[s]['test_accuracy']))
        accuracy_std.append(np.std(res[s]['test_accuracy']))

    plt.figure(figsize=(3, 8))
    plt.errorbar(accuracy_mean, sites, xerr=accuracy_std, fmt='o')
    plt.grid(linestyle='-', linewidth=.3)
    plt.show()

    return res


def load_data(filepath, drop=['SubjID', 'SubID', 'Age', 'Sex'], losocv=True):
    '''
    :param filepath: ends with .csv
    :param drop: all non data columns except site
    :param losocv: only include balanced sites for losocv / single site, otherwise False
    :return:
    '''

    df = pd.read_csv(filepath)
    df.rename(columns={'SITES': 'Site', 'SUBJECT_ID': 'SubjID'}, inplace=True, errors='ignore')

    sites = []

    # binary classification - PTSD/Control
    df = df[~df.Diagnosis.isnull()]
    df.loc[df.Diagnosis.isin(['TEHC', 'HC']), 'Diagnosis'] = 'Control'

    # for losocv, single site analysis
    if losocv:
        # only include sites with > 10 PTSD and 10 Control samples
        if 'Site' in df:
            for s in df.Site.unique():
                if all(x in df[df.Site == s].Diagnosis.unique() for x in ['PTSD', 'Control']):
                    if (df[df.Site == s].Diagnosis.value_counts()['PTSD'] > 10):
                        if df[df.Site == s].Diagnosis.value_counts()['Control'] > 10:
                            sites.append(s)
        else:
            print('need site information to run LOSOCV')
        df = df[df.Site.isin(sites)]
    else:
        drop.append('Site')

    final = df.drop(columns=drop, errors='ignore')
    final['Diagnosis'] = pd.Categorical(final.Diagnosis, categories=['Control', 'PTSD']).codes

    return final


def print_auc(test_y, pred_y):
    fpr, tpr, thr = roc_curve(test_y, pred_y)
    aucscore = auc(fpr, tpr)
    print('test auc: ', aucscore)
    plt.title('ROC curve', fontsize=15)
    plt.plot(fpr, tpr, label='AUC = %0.2f' % aucscore)
    plt.legend(loc='lower right', prop={'size': 15})
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.show()

def print_result(ytest, ypred):
    '''
    :param ytest: observed
    :param ypred: predicted
    :prints accuracy, sensitivity, specificity
    '''
    print('test accuracy: ', accuracy_score(ytest, ypred))
    tn, fp, fn, tp = confusion_matrix(ytest, ypred).ravel()
    print('TN: ', tn)
    print('FP: ', fp)
    print('FN: ', fn)
    print('TP: ', tp)
    print('sensitivity: ', tp / (tp + fn))
    print('specificity: ', tn / (tn + fp))


def impute_scale(xtrain, xtest):
    '''
    :param xtrain:
    :param xtest:
    :return: xtrain xtest with imputed mean and scale
    '''
    # drop non data column
    xtrain.drop(columns='Site', inplace=True, errors='ignore')
    xtest.drop(columns='Site', inplace=True, errors='ignore')

    # impute missing values with mean
    xtrain.fillna(xtrain.mean(), inplace=True)
    xtest.fillna(xtrain.mean(), inplace=True)

    # scale using sklearn robustscaler, keep column names
    cols = xtrain.columns.values
    transformer = RobustScaler().fit(xtrain)
    xtrain = transformer.transform(xtrain)
    xtest = transformer.transform(xtest)
    xtrain = pd.DataFrame(xtrain)
    xtest = pd.DataFrame(xtest)
    xtrain.columns = cols
    xtest.columns = cols

    return xtrain, xtest


# Leave one site out cross validation

def loso_cv_gridsearch(X, y, model, space):
    '''
    leave-one-site-out cross validation, plots averaged roc curve
    :param X:
    :param y: must have no missing values
    :param model: RandomForestClassifier() or SVC()
    :param space: parameter grid
    :return:
    '''

    model.probability = True
    tprs = []
    base_fpr = np.linspace(0, 1, 101)

    logo = LeaveOneGroupOut()
    grp = X['Site']

    for train_ix, test_ix in logo.split(X, y, groups=grp):
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

        print('left out site: ', X_test['Site'].unique())

        X_train, X_test = impute_scale(X_train, X_test)

        # Cross validation results
        AUCscore = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=StratifiedKFold(10), n_jobs=-1)
        ACCscore = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=StratifiedKFold(10), n_jobs=-1)
        print("CV Accuracy:", ACCscore.mean(), 'CV ACC std:', np.std(ACCscore))
        print("CV AUC:", AUCscore.mean(), 'CV AUC std: ', np.std(AUCscore))

        # Gridsearch to find best model
        search = GridSearchCV(model, space, scoring='accuracy', cv=StratifiedKFold(10), refit=True)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)

        # test acc, auc, sensitivity, auc
        print('before adjusting threshold: ')
        print_result(y_test, y_pred)
        y_hat = best_model.predict_proba(X_test)[:, 1]
        print_auc(y_test, y_hat)

        # adjust threshold
        fpr, tpr, thresholds = roc_curve(y_test, y_hat)
        gmeans = sqrt(tpr * (1 - fpr))
        # locate the index of the largest g-mean
        ix = argmax(gmeans)
        print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
        best_threshold = thresholds[ix]

        y_pred_class = binarize([y_hat], threshold=best_threshold)[0]
        print("After adjusting the proba treshold: ", best_threshold)
        print_result(y_test, y_pred_class)

        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    plt.plot(base_fpr, mean_tprs)
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.5)

    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# basic ML classification

def run_analysis(X, y, models):
    '''
    :param X: data matrix67
    :param y: label
    :param models: dictionary of models
    :return:
    '''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    X_train, X_test = impute_scale(X_train, X_test)

    for model_name in models:
        print(model_name)
        model = models[model_name]

        # calibrate the prediction probability
        model.probability = True

        AUCscore = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=StratifiedKFold(10), n_jobs=-1)
        ACCscore = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=StratifiedKFold(10), n_jobs=-1)
        print(model_name, "CV Accuracy:", ACCscore.mean(), 'CV ACC std:', np.std(ACCscore))
        print(model_name, "CV AUC:", AUCscore.mean(), 'CV AUC std: ', np.std(AUCscore))

        # test accuracy
        model.fit(X_train, y_train)
        print('train acc', model.score(X_train, y_train))

        # test accuracy
        y_pred = model.predict(X_test)
        print_result(y_test, y_pred)

        # Optimize predict probability threshold
        y_hat = model.predict_proba(X_test)[:, 1]
        print_auc(y_test, y_hat)
        fpr, tpr, thresholds = roc_curve(y_test, y_hat)

        print('test auc', auc(fpr, tpr))

        # calculate the g-mean for each threshold
        gmeans = sqrt(tpr * (1 - fpr))
        # locate the index of the largest g-mean
        ix = argmax(gmeans)
        print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
        best_threshold = thresholds[ix]

        y_pred_class = binarize([y_hat], threshold=best_threshold)[0]
        print("After adjusting the proba treshold: ", best_threshold)
        print_result(y_test, y_pred_class)


if __name__ == '__main__':
    dti = load_data('dti_with_controltype_20210622.csv', losocv=True)
    t1 = load_data('T1_Xin_fixed_v18c_cleaned_reduced.csv', losocv=False)

    # leave one Site out CV
    model = RandomForestClassifier(random_state=1)
    space = dict()
    space['n_estimators'] = [900, 600, 100]
    space['max_depth'] = [3, 5]
    space['criterion'] = ['gini', 'entropy']
    loso_cv_gridsearch(dti.drop(columns='Diagnosis'), dti['Diagnosis'], model, space)

    # single site
    stats = single_site_analysis(SVC(), dti)

    # overall ML
    models = {
        'RandomForest_default': RandomForestClassifier(),
        'SVC_default': SVC()
    }

    run_analysis(t1.drop(columns=['Diagnosis']), t1['Diagnosis'], models)

