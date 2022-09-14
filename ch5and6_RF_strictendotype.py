# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import matplotlib
#matplotlib.use('Agg')
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict,cross_val_score
from sklearn.feature_selection import SelectPercentile, f_classif,SelectFromModel, VarianceThreshold, RFECV
from sklearn.preprocessing import StandardScaler,MaxAbsScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, classification_report, f1_score
import matplotlib.pyplot as plt
from imblearn.ensemble import *
from imblearn.metrics import classification_report_imbalanced
from sklearn.ensemble import RandomForestClassifier


data=([line.rstrip("\n").split("\t") for line in open("../GENEPY_JULY21_PHRED15.matrix",'r')])
#data=([line.rstrip("\n").split("\t") for line in open("../GENEPY_JUNE2021.matrix",'r')])
data = np.array(data)
data[data=="NA"]= np.nan


metadata = np.array(([line.rstrip("\n").split("\t") for line in open("../20210427_metadata.txt",'r')]))
#metadata = np.array(([line.rstrip("\n").split("\t") for line in open("../Metadata.txt",'r')]))

genes = data[0,1:]
y = metadata[1:,12]


X = data[1:,1:].astype(float)

#metadata[1:,6][metadata[1:,6]=="."] = np.nan
metadata[1:,3][metadata[1:,3]=="."] = np.nan
#metadata[1:,7][metadata[1:,7]=="."] = np.nan

_ethmask = ((metadata[1:,2]=="EUR") & \
            (metadata[1:,3].astype(float)>0.9) & \
            (metadata[1:,11]=="CD") & \
            (metadata[1:,7]=="0"))

X = X[_ethmask,:]
y = y[_ethmask]

### Which groups to select?
g1="1"
g2="0"


_indv_mask= np.zeros(len(y)).astype(bool)
_indv_mask[y==g1]=True
_indv_mask[y==g2]=True

y=y[_indv_mask]
X = X[_indv_mask,:]


y[y=="1"]="Stricturing"
y[y=="0"]="Not-stricturing"
g1 = "Stricturing"
g2 = "Not-stricturing"

### Filter by Fuentes

fuentes = np.array(([line.rstrip() for line in open("C:/Users/iss1g18/OneDrive - University of Southampton/Desktop/ML_GENEPY/202111_nofuentes_genepyphred.txt","r")]))

_fuentesmask=np.zeros(len(genes)).astype(bool)

for i,n in enumerate(genes):
    if n in fuentes:
        _fuentesmask[i]=True

genes = genes[_fuentesmask]
X=X[:,_fuentesmask]


##Panel

#panel = np.array(([line.rstrip() for line in open("..20220106_strict_panel_exclusive.txt","r")]))
#panel = np.array(([line.rstrip() for line in open("../20211004_stricturing_genes_inclusive.txt","r")]))
#panel = np.array(([line.rstrip() for line in open("../HTG_seq_AI.txt","r")]))
#panel = np.array(([line.rstrip() for line in open("../20211004_ibd_monogwas.txt","r")]))
#panel = np.array(([line.rstrip() for line in open("../20220107_keggnodsigpathway.txt","r")]))

#_panelmask= np.zeros(len(genes)).astype(bool)
#for i,n in enumerate(genes):
#    if n in panel:
#        _panelmask[i]=True

#genes = genes[_panelmask]
#X=X[:,_panelmask]


vt =VarianceThreshold(threshold=0)
vt.fit(X)
X = vt.transform(X)
genes=vt.transform(genes.reshape(1,-1))

sel_ge = genes[0]

#%% SCALING
#ss = StandardScaler()
ss = MaxAbsScaler()
#ss= MinMaxScaler()
X = ss.fit_transform(X)

#%% Data Split
if g2=="CD":
    _trainsize=150 #set to 150 if CD
else:
    _trainsize=136

np.random.seed(42)
_controls = np.random.choice(np.where(y==g1)[0],size=_trainsize, replace=False)
np.random.seed(42)
_ibd = np.random.choice(np.where(y==g2)[0],size=_trainsize, replace=False,)
#
_index = np.hstack((_controls,_ibd))
_mask = np.zeros(len(y),dtype=bool)
_mask[_index]=True
X_train = X[_mask,:]
y_train = y[_mask]
X_test = X[~_mask,:]
y_test = y[~_mask]


#%% Dataset Sizes
from collections import Counter
print("FULL ", Counter(y))
print("TRAIN ",Counter(y_train))
print("TEST ",Counter(y_test))


#%% Feature Selection

from sklearn.model_selection import StratifiedKFold, LeaveOneOut, LeavePOut
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
#


#
w =[]
train_acc = []
skf = StratifiedKFold(n_splits=5)
#skf = LeaveOneOut()
#skf = LeavePOut(p=5)
for train_index, test_index in skf.split(X_train, y_train):
    
    Xa, Xb = X_train[train_index], X_train[test_index]
    ya, yb = y_train[train_index], y_train[test_index]
    
#    anova_filter = SelectKBest(f_classif, k=1000)
    clf = LinearSVC(C=1, penalty="l1", dual=False,max_iter=1e5)
#    model = make_pipeline(anova_filter, clf)
    clf.fit(Xa, ya==g1)

    w.append(clf.coef_)

print('Training 5-fold  F1: %.2f +- %.2f' %(np.mean(train_acc), np.std(train_acc)))
_mask = np.any(w, axis=0)[0]
print('Feature selected: %.i' %(sum(_mask)))
X_train = X_train[:,_mask]
X_test = X_test[:,_mask]
sel_ge = sel_ge[_mask]

#%% Manual Nested CV

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from imblearn.metrics import *
from sklearn.metrics import * 
import numpy as np
from skopt import BayesSearchCV

#Dataset Split

X_1 = X_train[0:38,:]
X_2 = X_train[39:77,:]
X_3 = X_train[78:116,:]
X_4 = X_train[117:155,:]
X_5 = X_train[156:194,:]
X_6 = X_train[195:233,:]
X_7 = X_train[234:272,:]

y_1 = y_train[0:38,]
y_2 = y_train[39:77,]
y_3 = y_train[78:116,]
y_4 = y_train[117:155,]
y_5 = y_train[156:194,]
y_6 = y_train[195:233,]
y_7 = y_train[234:272,]

#Prep Gridsearch Datasets

X_1_train_hp = np.concatenate([X_1, X_2, X_3, X_4, X_5, X_6])
X_2_train_hp = np.concatenate([X_1, X_2, X_3, X_4, X_5, X_7])
X_3_train_hp = np.concatenate([X_1, X_2, X_3, X_4, X_6, X_7])
X_4_train_hp = np.concatenate([X_1, X_2, X_3, X_5, X_6, X_7])
X_5_train_hp = np.concatenate([X_1, X_2, X_4, X_5, X_6, X_7])
X_6_train_hp = np.concatenate([X_1, X_3, X_4, X_5, X_6, X_7])
X_7_train_hp = np.concatenate([X_2, X_3, X_4, X_5, X_6, X_7])

y_1_train_hp = np.concatenate([y_1, y_2, y_3, y_4, y_5, y_6])
y_2_train_hp = np.concatenate([y_1, y_2, y_3, y_4, y_5, y_7])
y_3_train_hp = np.concatenate([y_1, y_2, y_3, y_4, y_6, y_7])
y_4_train_hp = np.concatenate([y_1, y_2, y_3, y_5, y_6, y_7])
y_5_train_hp = np.concatenate([y_1, y_2, y_4, y_5, y_6, y_7])
y_6_train_hp = np.concatenate([y_1, y_3, y_4, y_5, y_6, y_7])
y_7_train_hp = np.concatenate([y_2, y_3, y_4, y_5, y_6, y_7])

X_1_test = X_7
X_2_test = X_6
X_3_test = X_5
X_4_test = X_4
X_5_test = X_3
X_6_test = X_2
X_7_test = X_1

y_1_test = y_7
y_2_test = y_6
y_3_test = y_5
y_4_test = y_4
y_5_test = y_3
y_6_test = y_2
y_7_test = y_1

#Grid Search Variables

params = dict()
params['min_samples_leaf'] = (1,2,3,4,5,6)
params['max_features'] = ('sqrt', 'log2', None)
params['min_samples_split'] = (2,3,4,5,6)
params['max_depth'] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30, None]
params['n_estimators'] = [100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

rfc = RandomForestClassifier(bootstrap=True, oob_score=True,random_state=42)

# define evaluation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
#cv = RepeatedStratifiedKFold(n_splits=7, n_repeats=3, random_state=1)
# define the search
search = BayesSearchCV(estimator=rfc, search_spaces=params, n_jobs=-1, cv=cv, verbose=1, scoring='balanced_accuracy', n_iter=60)
# perform the search
result = search.fit(X_7_train_hp, y_7_train_hp)
# report the best result
print(search.best_score_)
print(search.best_params_)

# Get generalised CV

best_model_7 = result.best_estimator_
# evaluate model on the hold out dataset
y_7_hat = best_model_7.predict(X_7_test)
# evaluate the model
bal_acc_7 = balanced_accuracy_score(y_7_test, y_7_hat)
# report progress
print(bal_acc_7)


#%% Individual Hyperparameter Trials - Not needed to run ML pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import numpy as np
import seaborn as sbn

minsamplesleaf = [1,2,3,4,5,6,7,8,9,10]
maxfeatures = ["sqrt", "log2", None]
minsamplessplit = [2,3,4,5,6,7,8,9,10]
maxdepth = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30, None]
nestimators = [100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
#p_grid = {"min_samples_leaf": [1,2,4]}

rfc = RandomForestClassifier(bootstrap=True, oob_score=True,random_state=42)

inner_cv = KFold(n_splits=7, shuffle=True, random_state=42)

clf = GridSearchCV(estimator=rfc, param_grid=dict(min_samples_split=minsamplessplit), cv=inner_cv, verbose=3, scoring='balanced_accuracy')
clf.fit(X_train, y_train)

grid_scores = clf.cv_results_ 



def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
         ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))
         
    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    
plot_grid_search(clf.cv_results_, maxdepth, maxfeatures, 'Max Depth', 'Max Features')

# One variable

    for idx, val in enumerate(grid_param_2):
         ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))
         
         
    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')

# Calling Method 
plot_grid_search(clf.cv_results_, maxdepth, maxfeatures, 'Max Depth', 'Max Features')

## Bar charts

#Max Depth
Maximum_Depth = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','None']

def plot_grid_search(cv_results, grid_param_1, name_param_1):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_1))
    Maximum_Depth = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22'
                     ,'23','24','25','26','27','28','29','30', 'None']
    
    # Plot Grid search scores
    _, ax = plt.subplots(1,1, figsize=(11,5))

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    ax.bar(grid_param_1, scores_mean, color='c')
         
    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    #ax.grid('on', color='lightgrey')
    
plot_grid_search(clf.cv_results_, Maximum_Depth, 'Maximum Depth')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
Maximum_Depth = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22'
                 ,'23','24','25','26','27','28','29','30']
scores_mean = grid_scores['mean_test_score']
scores_mean = np.array(scores_mean).reshape(len(maxdepth))
ax.bar(Maximum_Depth,scores_mean)
plt.show()

# Maximum Depth - LINE
def plot_grid_search(cv_results, grid_param_1, name_param_1):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1,figsize=(6,5))

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    ax.plot(grid_param_1, scores_mean, color='c')
         
    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.grid('on')
    
plot_grid_search(clf.cv_results_, nestimators, 'Number of Estimators')


# Max Features
Maximum_Features = ['sqrt', 'log2', 'None']

def plot_grid_search(cv_results, grid_param_1, name_param_1):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_1))
    Maximum_Features = ['sqrt', 'log2', 'None']
    
    # Plot Grid search scores
    _, ax = plt.subplots(1,1, figsize=(6,5))

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    ax.bar(grid_param_1, scores_mean, color='c')
         
    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.grid('on', color='lightgrey')
    
plot_grid_search(clf.cv_results_, Maximum_Features, 'Max Features')


# Number of Estimators - BAR
N_estimators = ['100', '250', '500', '750', '1000', '5000', '10000']

def plot_grid_search(cv_results, grid_param_1, name_param_1):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_1))
    N_estimators = ['100', '250', '500', '750', '1000', '5000', '10000']
    
    # Plot Grid search scores
    _, ax = plt.subplots(1,1, figsize=(6,5))

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    ax.bar(grid_param_1, scores_mean, color='c')
         
    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.grid('on', color='lightgrey')
    
plot_grid_search(clf.cv_results_, N_estimators, 'Number of Estimators')


# Number of Estimators - LINE
def plot_grid_search(cv_results, grid_param_1, name_param_1):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1,figsize=(6,5))

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    ax.plot(grid_param_1, scores_mean, color='c')
         
    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.grid('on')
    
plot_grid_search(clf.cv_results_, nestimators, 'Number of Estimators')


# Min Samples Leaf - LINE
def plot_grid_search(cv_results, grid_param_1, name_param_1):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1,figsize=(6,5))

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    ax.plot(grid_param_1, scores_mean, color='c')
         
    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.grid('on')
    
plot_grid_search(clf.cv_results_, minsamplesleaf, 'Minimum Samples per Leaf')


# Min Samples Split - LINE
def plot_grid_search(cv_results, grid_param_1, name_param_1):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1,figsize=(6,5))

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    ax.plot(grid_param_1, scores_mean, color='c')
         
    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.grid('on')
    
plot_grid_search(clf.cv_results_, minsamplessplit, 'Minimum Samples per Split')


#%% Random Forest
print('#################################\n Random Forest \n#################################')
#

#clf = RandomForestClassifier(n_estimators=10000, oob_score=True,random_state=42)
#clf = RUSBoostClassifier(n_estimators=10000,random_state=42)
#clf = EasyEnsembleClassifier(n_estimators=10000, random_state=42)

from sklearn.model_selection import GridSearchCV

clf = RandomForestClassifier(n_estimators=10000, oob_score=True,random_state=42, max_depth = None, max_features = "sqrt", 
                                 min_samples_leaf= 1, min_samples_split= 2)

clf.fit(X_train, y_train)
try:
    print('OOB on Train set: %.3f' %(clf.oob_score_))
except:
    print('Estimator error (1-E)  on Train set: %.3f' %(1-np.mean(clf.estimator_errors_)))
#
y_pred = clf.predict(X_test)
predicted_probas = clf.predict_proba(X_test)
print(classification_report_imbalanced(y_test,y_pred))


print("AUROC on Test set: %.3f" %(roc_auc_score(y_test,predicted_probas[:,1])))


#%% Plot AUC, Confusion Matrix
import scikitplot as skplt
import seaborn as sbn

skplt.metrics.plot_roc(y_test,predicted_probas, plot_macro=False, plot_micro=False, classes_to_plot="Not-stricturing")
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)

#skplt.estimators.plot_learning_curve(clf, X_train, y_train, cv=5, scoring='roc_auc', train_sizes=np.linspace(.1,1.0,5), shuffle=True)

skplt.estimators.plot_feature_importances(clf, feature_names=sel_ge, x_tick_rotation=30, text_fontsize=8, max_num_features=10)


#%% Feature Importances
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
ordered_feature_names = np.array(sel_ge)[indices]
dd =dict(np.vstack((sel_ge,importances)).T)

#%% Mann Whitney and Gene Distributions
import seaborn as sbn
import pandas as pd
pid = data[1:,0][_ethmask][_indv_mask]
g=pd.DataFrame(X[:,:])
#g.columns=sel_ge
g.columns=genes[0]
g["SamID"]=pid

g["group"]=y
#pp = sbn.pairplot(data=g, vars=ordered_feature_names[:10], hue="group")
fig = plt.figure(figsize=(10,8))

for i,n in enumerate(ordered_feature_names[:10]):
    ax = fig.add_subplot(5,2,i+1)
    sbn.kdeplot(g[n][g['group']==g1], label=g1, shade=True)
    sbn.kdeplot(g[n][g['group']==g2], label=g2,shade=True)
    ax.set_xlabel(n)
fig.tight_layout()

print(", ".join(ordered_feature_names[:10]))

from scipy.stats import mannwhitneyu as mu
egg =[]
for n in ordered_feature_names[:]:
    try:
        pval=mu(g[n][g['group']==g1],g[n][g['group']==g2])[1]
        egg.append([n,pval,dd[n]])
    except:
        egg.append([n,1,dd[n]])
egg= np.array(egg)  
np.savetxt("../output_genes.txt",egg,fmt="%s")

#%% violin
plt.figure()
df1 = g[ordered_feature_names[:10]][g['group']==g1].assign(Group=g1)
df2 = g[ordered_feature_names[:10]][g['group']==g2].assign(Group=g2)
cdf = pd.concat([df1,df2])
mdf = pd.melt(cdf, id_vars=['Group'])
ax = sbn.violinplot(x="variable",y="value", hue="Group", data=mdf)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)

#%% SAVE FIGS
from matplotlib.backends.backend_pdf import PdfPages

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
multipage("../output_figures.pdf")

