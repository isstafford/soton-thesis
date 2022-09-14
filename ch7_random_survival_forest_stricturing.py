# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 16:25:58 2022

@author: iss1g18
"""

#%% Load in Packages

import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np
import seaborn as sbn

from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, KFold, RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, LeavePOut

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler,MaxAbsScaler, MinMaxScaler

from sksurv.datasets import load_gbsg2
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import *
from sksurv.metrics import *

from sklearn.feature_selection import SelectPercentile, f_classif, SelectFromModel, VarianceThreshold, RFECV
from sklearn.metrics import *
from imblearn.ensemble import *
from imblearn.metrics import *

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from skopt import BayesSearchCV
from matplotlib.backends.backend_pdf import PdfPages

from sklearn import linear_model


#%% Data Pre-Processing

data=([line.rstrip("\n").split("\t") for line in open("../GENEPY_JULY21_PHRED15.matrix",'r')])
data = np.array(data)
data[data=="NA"]= np.nan


metadata = np.array(([line.rstrip("\n").split("\t") for line in open("../202201_metadata_edit.txt",'r')]))
#metadata = np.array(([line.rstrip("\n").split("\t") for line in open("../Metadata.txt",'r')]))

genes = data[0,1:]
y = metadata[1:,np.r_[12,15]] #Do I need to alter to include follow-up time??

X = data[1:,1:].astype(float)

metadata[1:,3][metadata[1:,3]=="."] = np.nan
#metadata[1:,15][metadata[1:,15]==.] = np.nan

_ethmask = ((metadata[1:,2]=="EUR") & \
            (metadata[1:,3].astype(float)>0.9) & \
            (metadata[1:,11]=="CD") & \
            (metadata[1:,7]=="0") & \
            (metadata[1:,15]!="."))

X = X[_ethmask,:]
y = y[_ethmask,]

### Which groups to select?
g1="1"
g2="0"

_indv_mask= np.zeros(len(y)).astype(bool)
_indv_mask[y[:,0]==g1]=True
_indv_mask[y[:,0]==g2]=True

y=y[_indv_mask]
X = X[_indv_mask,:]

y[:,0] = np.where((y[:,0]=="1"), "Stricturing",y[:,0])
y[:,0] = np.where((y[:,0]=="0"), "Not-stricturing",y[:,0])

g1 = "Stricturing"
g2 = "Not-stricturing"

### Filter by Fuentes

fuentes = np.array(([line.rstrip() for line in open("../202111_nofuentes_genepyphred.txt","r")]))

_fuentesmask=np.zeros(len(genes)).astype(bool)

for i,n in enumerate(genes):
    if n in fuentes:
        _fuentesmask[i]=True

genes = genes[_fuentesmask]
X=X[:,_fuentesmask]


##Panel

#panel = np.array(([line.rstrip() for line in open("../20220106_strict_panel_exclusive.txt","r")]))
panel = np.array(([line.rstrip() for line in open("../20211004_stricturing_genes_inclusive.txt","r")]))
#panel = np.array(([line.rstrip() for line in open("../HTG_seq_AI.txt","r")]))
#panel = np.array(([line.rstrip() for line in open("../20211004_ibd_monogwas.txt","r")]))
#panel = np.array(([line.rstrip() for line in open("../20220107_keggnodsigpathway.txt","r")]))
#panel = np.array(([line.rstrip() for line in open("../top10featselrsf.txt","r")]))
#panel = np.array(([line.rstrip() for line in open("../bottom10featselrsf.txt","r")]))
#panel = np.array(([line.rstrip() for line in open("../NOD2.txt","r")]))


_panelmask= np.zeros(len(genes)).astype(bool)
for i,n in enumerate(genes):
    if n in panel:
        _panelmask[i]=True

genes = genes[_panelmask]
X=X[:,_panelmask]


vt =VarianceThreshold(threshold=0)
vt.fit(X)
X = vt.transform(X)
genes=vt.transform(genes.reshape(1,-1))

sel_ge = genes[0]


#%% Pre-processing for feature selection 

## Using binomial probability to ensure no invariant genes in the training set ##
## (condition for the use of CPH as feature ranking) ##

from scipy.stats import binom

####################################################################
###### PROBABILITY CALCULATION IF WORKING FROM FULL DATASET X ######
####################################################################

## Find the proportion of each gene
#ngenes = len(X[0])
#nz = X.astype(bool).sum(axis=0) # num people with nonzero values in full dataset
#known_prop = np.zeros(shape=(ngenes))
##for j in range(0,ngenes,1):
##   known_prop[j,] = nz[j,]/len(X)

#known_prop = nz/len(X)

## var known_prop is now vector that contains the proportion of people per gene that have GenePy scores > 0
#known_prop = np.delete(known_prop,np.where(nz<=5))
##Probability function

#_trainset = 244
#probability = (1 - binom.pmf(0,_trainset,known_prop))
#nzdataset_prob=np.prod(probability, axis=0) #probability that all columns meet the condition score>0 with 95% probability

#known_prop = np.delete(known_prop_s,np.asarray(np.argmin(known_prop)))

##Produce non-zero dataset

#while np.prod((1 - binom.pmf(0,_trainset,known_prop)),axis=0) < 0.95:
#    known_prop = np.delete(known_prop,np.asarray(np.argmin(known_prop))) # Remove min probability from known_prop
#    sel_ge = np.delete(sel_ge,np.argmin(known_prop)) # Remove corresponding gene (gene name list is separate)

####################################################################
####################################################################

#############################################################################
###### PROBABILITY CALCULATION IF CHOOSING FROM TWO CLASSES SEPARATELY ######
#############################################################################

## Split X into stricturing and not-stricturing
s_ind = np.where(y[:,0]=="Stricturing")
num_strict = len(s_ind[0])
s_ind = np.asarray(s_ind)
s_ind = s_ind.reshape(num_strict)
X_s =  X[s_ind,:]

ns_ind = np.where(y[:,0]=="Not-stricturing")
num_notstrict = len(ns_ind[0])
ns_ind = np.asarray(ns_ind)
ns_ind = ns_ind.reshape(num_notstrict)
X_ns =  X[ns_ind,:]

ngenes = len(X[0])

_trainsize = 122
sel_ge_s = sel_ge

## STRICTURING DATA CALCULATION

nz_s = X_s.astype(bool).sum(axis=0) # num people with nonzero values in X_s
known_prop_s = np.zeros(shape=(ngenes))
known_prop_s = nz_s/len(X_s)

## NOT STRICTURING DATA CALCULATION

nz_ns = X_ns.astype(bool).sum(axis=0) # num people with nonzero values in X_s
known_prop_ns = np.zeros(shape=(ngenes))
known_prop_ns = nz_ns/len(X_ns)

#known_prop_ns contains the proportion of people in not-stricturing class per gene that have GenePy scores > 0

# Probability calculation - not-stricturing

_trainsize = 122
sel_ge_ns = sel_ge

known_prop_all = np.add(known_prop_s, known_prop_ns)


while np.prod((1 - binom.pmf(0,_trainsize,known_prop_s)),axis=0)*np.prod((1 - binom.pmf(0,_trainsize,known_prop_ns)),axis=0) < 0.99:
    min_prop = np.argmin(known_prop_all)
    #print(min_prop)
    known_prop_all = np.delete(known_prop_all,np.asarray(min_prop))#probability that all columns meet the condition score>0 with 95% probability
    known_prop_s = np.delete(known_prop_s,np.asarray(min_prop)) # Remove min probability from known_prop
    sel_ge_s = np.delete(sel_ge_s,min_prop) # Remove corresponding gene (gene name list is separate)
    known_prop_ns = np.delete(known_prop_ns,np.asarray(min_prop)) # Remove min probability from known_prop


## FINAL DATASET ASSEMBLY 

#Get indices of the gene overlap

goverlap, g_ind, g_ind_null = np.intersect1d(sel_ge,sel_ge_s, return_indices=True)

X = X[:, g_ind]
sel_ge = sel_ge[g_ind] 

#%% Random 10 Genes

np.random.seed(42)
randomgenes = np.random.choice(sel_ge, 10, replace = False)
rov, rand_ind, rand_ind_null = np.intersect1d(sel_ge,randomgenes, return_indices=True)

X = X[:, rand_ind]
sel_ge = sel_ge[rand_ind] 

#%% Data Split

_trainsize=122

np.random.seed(42)
_controls = np.random.choice(np.where(y==g1)[0],size=_trainsize, replace=False)
np.random.seed(42)
_ibd = np.random.choice(np.where(y==g2)[0],size=_trainsize, replace=False,)
#
_index = np.hstack((_controls,_ibd))
_mask = np.zeros(len(y),dtype=bool)
_mask[_index]=True
X_train = X[_mask,:]
y_train = y[_mask,:]
X_test = X[~_mask,:]
y_test = y[~_mask,:]


from collections import Counter
print("FULL ", Counter(y[:,0]))
print("TRAIN ",Counter(y_train[:,0]))
print("TEST ",Counter(y_test[:,0]))

X_train.astype(bool).sum(axis=0)

#%% SCALING

#ss = StandardScaler()
ss = MaxAbsScaler()
#ss= MinMaxScaler()
#fit on train here ##
ss_fit = ss.fit(X_train)
X_train = ss_fit.transform(X_train)
X_test = ss_fit.transform(X_test)
#X = X_temp + 1

#%% PCA Feature Selection

from sklearn.decomposition import PCA
from kneed import KneeLocator
import matplotlib.pyplot as plt
import pandas as pd

pca = PCA(0.95)
#pca = PCA(n_components=41)

modelpca=pca.fit(X_train)

components = modelpca.n_components_

print(components)

expvar = modelpca.explained_variance_ratio_
expvar=pd.Series(expvar).sort_values(ascending=False)
expvar = list(np.float_(expvar))

x =range(len(expvar))

"""Elbow Calculation"""
kn = KneeLocator(x,expvar,curve='convex', direction='decreasing', interp_method='interp1d')
print(kn.knee)

plt.xlabel('PCs')
plt.ylabel('% Explained Variance')
plt.xticks(range(0,297,5))
plt.plot(x, expvar, '-')
f=plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')

#PC loadings

featnames = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10',
             'PC11','PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19',
             'PC20','PC21','PC22','PC23','PC24','PC25','PC26','PC27','PC28',
             'PC29','PC30','PC31','PC32','PC33','PC34']

featnames = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9']

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_matrix = pd.DataFrame(loadings, columns=featnames, index=sel_ge)
loading_matrix

#Transform data

X_train = modelpca.transform(X_train)
X_test = modelpca.transform(X_test)

#%% y dataset modification for rsf

##Save for later


y_train[:,0] = np.where((y_train[:,0]=="Stricturing"), "1",y_train[:,0])
y_train[:,0] = np.where((y_train[:,0]=="Not-stricturing"), "0" ,y_train[:,0])

y_test[:,0] = np.where((y_test[:,0]=="Stricturing"), "1",y_test[:,0])
y_test[:,0] = np.where((y_test[:,0]=="Not-stricturing"), "0",y_test[:,0])

y_train = np.core.records.fromarrays(y_train.transpose(), names='col1, col2', formats = 'bool, f8') ###
y_test = np.core.records.fromarrays(y_test.transpose(), names='col1, col2', formats = 'bool, f8') ###

#y[:,0] = np.where((y[:,0]=="Stricturing"), "1",y[:,0])
#y[:,0] = np.where((y[:,0]=="Not-stricturing"), "0" ,y[:,0])
#y = np.core.records.fromarrays(y.transpose(), names='col1, col2', formats = 'bool, f8')



#%% Feature Selection

from sklearn import utils
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline


import numpy as np

#X_train_sub = X_train[:,78:81]
#sel_ge_sub = sel_ge[78:81,]

def fit_and_score_features(X_frame, y_frame):
    n_features = X_frame.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis()
    for j in range(n_features):
        Xj = X_frame[:, j:j+1]
        m.fit(Xj, y_frame)
        scores[j] = m.score(Xj, y_frame)
    return scores

#### 50 iters - stricturing panel #######
per = 220


X_100, y_100 = utils.resample(X_train, y_train, n_samples = per, random_state=100)
scores = fit_and_score_features(X_100, y_100)
feature_order_100 = pd.Series(scores, index=sel_ge).sort_values(ascending=False)

################


#ind = range(1,74,1)

############# GridSearch and Bayes Search trials


per = 220
rand = 1
iters=30



X_50, y_50 = utils.resample(X_train, y_train, n_samples = per, random_state=50)
scores = fit_and_score_features(X_50, y_50)
feature_order_50 = pd.Series(scores, index=sel_ge).sort_values(ascending=False)
pd.Series(scores, index=sel_ge).sort_values(ascending=False)
pipe = Pipeline([
                    ('select', SelectKBest(fit_and_score_features, k=3)),
                     ('model', CoxPHSurvivalAnalysis())])

param_grid = {'select__k': np.arange(1, X_50.shape[1] + 1)}
cv = KFold(n_splits=3, random_state=rand, shuffle=True)
gcv = GridSearchCV(pipe, param_grid, return_train_score=True, cv=cv, verbose=3)
#gcv = BayesSearchCV(pipe, param_grid, return_train_score=True, cv=cv, verbose=3, n_iter=iters)
gcv.fit(X_50,y_50)
results = pd.DataFrame(gcv.cv_results_).sort_values(by='mean_test_score', ascending=False)
results.loc[:, ~results.columns.str.endswith("_time")]

pipe.set_params(**gcv.best_params_)
pipe.fit(X_50, y_50)

transformer, final_estimator = [s[1] for s in pipe.steps]
features_sel = pd.Series(final_estimator.coef_, index=sel_ge[transformer.get_support()])
features_sel = np.asarray(sel_ge[transformer.get_support()])

fsel, fsel_ind, fsel_null = np.intersect1d(sel_ge,features_sel, return_indices=True)

sel_ge_50 = sel_ge[fsel_ind]
X_50 = X_50[:,fsel_ind]

print(len(sel_ge_50))
#######################################


#feature_order_df = pd.DataFrame.from_items(zip(series.index, series.str.split('\t'))).T

#feature_order.to_txt("C:/Users/iss1g18/OneDrive - University of Southampton/Documents/ML_IBD_2021/rsf/feature_ranking_all_data.txt", index = False)

#X_train.astype(bool).sum(axis=0)



pipe = Pipeline([
                 ('select', SelectKBest(fit_and_score_features, k=3)),
                 ('model', CoxPHSurvivalAnalysis())])

param_grid = {'select__k': np.arange(1, X_train.shape[1] + 1)}
cv = KFold(n_splits=3, random_state=1, shuffle=True)
#gcv = GridSearchCV(pipe, param_grid, return_train_score=True, cv=cv, verbose=3)
gcv = BayesSearchCV(pipe, param_grid, return_train_score=True, cv=cv, verbose=3, n_iter=30)
gcv.fit(X_train,y_train)
#best_model = fit.best_estimator_
#best_params = fit.best_params_

#X_train_trial = fit.fit_transform(X_train, y_train)
#fit.get_support()

results = pd.DataFrame(gcv.cv_results_).sort_values(by='mean_test_score', ascending=False)
results.loc[:, ~results.columns.str.endswith("_time")]

pipe.set_params(**gcv.best_params_)
pipe.fit(X_train, y_train)

transformer, final_estimator = [s[1] for s in pipe.steps]
features_sel = pd.Series(final_estimator.coef_, index=sel_ge[transformer.get_support()])
features_sel = np.asarray(sel_ge[transformer.get_support()])

fsel, fsel_ind, fsel_null = np.intersect1d(sel_ge,features_sel, return_indices=True)

sel_ge = sel_ge[fsel_ind]
X_train = X_train[:,fsel_ind]
X_test = X_test[:,fsel_ind]

topg=([line.rstrip("\n").split("\t") for line in open("C:/Users/iss1g18/OneDrive - University of Southampton/Documents/ML_IBD_2021/rsf/topg_knee_ibdpanel.txt",'r')])
topg = np.asarray(topg)
fsel, fsel_ind, fsel_null = np.intersect1d(sel_ge,topg, return_indices=True)
sel_ge = sel_ge[fsel_ind]
X_train = X_train[:,fsel_ind]
X_test = X_test[:,fsel_ind]

#%% Manual Nested CV
X_hyper = X_train
y_hyper=y_train

#Dataset Split
_bayessize=24

np.random.seed(42)
_ns_1 = np.random.choice(np.where(y_hyper==g2)[0],size=_bayessize, replace=False)
np.random.seed(42)
_s_1 = np.random.choice(np.where(y_hyper==g1)[0],size=_bayessize, replace=False)
_index_1 = np.hstack((_ns_1,_s_1))
_mask_1 = np.zeros(len(y_hyper),dtype=bool)
_mask_1[_index_1]=True
X_1 = X_hyper[_mask_1,:]
y_1 = y_hyper[_mask_1,:]
X_hyper = X_hyper[~_mask_1,:]
y_hyper = y_hyper[~_mask_1,:]

np.random.seed(42)
_ns_2 = np.random.choice(np.where(y_hyper==g2)[0],size=_bayessize, replace=False)
np.random.seed(42)
_s_2 = np.random.choice(np.where(y_hyper==g1)[0],size=_bayessize, replace=False)
_index_2 = np.hstack((_ns_2,_s_2))
_mask_2 = np.zeros(len(y_hyper),dtype=bool)
_mask_2[_index_2]=True
X_2 = X_hyper[_mask_2,:]
y_2 = y_hyper[_mask_2,:]
X_hyper = X_hyper[~_mask_2,:]
y_hyper = y_hyper[~_mask_2,:]

np.random.seed(42)
_ns_3 = np.random.choice(np.where(y_hyper==g2)[0],size=_bayessize, replace=False)
np.random.seed(42)
_s_3 = np.random.choice(np.where(y_hyper==g1)[0],size=_bayessize, replace=False)
_index_3 = np.hstack((_ns_3,_s_3))
_mask_3 = np.zeros(len(y_hyper),dtype=bool)
_mask_3[_index_3]=True
X_3 = X_hyper[_mask_3,:]
y_3 = y_hyper[_mask_3,:]
X_hyper = X_hyper[~_mask_3,:]
y_hyper = y_hyper[~_mask_3,:]

_bayessize=25

np.random.seed(42)
_ns_4 = np.random.choice(np.where(y_hyper==g2)[0],size=_bayessize, replace=False)
np.random.seed(42)
_s_4 = np.random.choice(np.where(y_hyper==g1)[0],size=_bayessize, replace=False)
_index_4 = np.hstack((_ns_4,_s_4))
_mask_4 = np.zeros(len(y_hyper),dtype=bool)
_mask_4[_index_4]=True
X_4 = X_hyper[_mask_4,:]
y_4 = y_hyper[_mask_4,:]
X_5 = X_hyper[~_mask_4,:]
y_5 = y_hyper[~_mask_4,:]

#Prep Gridsearch Datasets

X_1_train_hp = np.concatenate([X_1, X_2, X_3, X_4])
X_2_train_hp = np.concatenate([X_1, X_2, X_3, X_5])
X_3_train_hp = np.concatenate([X_1, X_2, X_4, X_5])
X_4_train_hp = np.concatenate([X_1, X_3, X_4, X_5])
X_5_train_hp = np.concatenate([X_2, X_3, X_4, X_5])

y_1_train_hp = np.concatenate([y_1, y_2, y_3, y_4])
y_2_train_hp = np.concatenate([y_1, y_2, y_3, y_5])
y_3_train_hp = np.concatenate([y_1, y_2, y_4, y_5])
y_4_train_hp = np.concatenate([y_1, y_3, y_4, y_5])
y_5_train_hp = np.concatenate([y_2, y_3, y_4, y_5])

X_1_test = X_5
X_2_test = X_4
X_3_test = X_3
X_4_test = X_2
X_5_test = X_1

y_1_test = y_5
y_2_test = y_4
y_3_test = y_3
y_4_test = y_2
y_5_test = y_1

y_1_train_hp[:,0] = np.where((y_1_train_hp[:,0]=="Stricturing"), "1",y_1_train_hp[:,0])
y_1_train_hp[:,0] = np.where((y_1_train_hp[:,0]=="Not-stricturing"), "0" ,y_1_train_hp[:,0])
y_1_train_hp = np.core.records.fromarrays(y_1_train_hp.transpose(), names='col1, col2', formats = 'bool, f8')

y_2_train_hp[:,0] = np.where((y_2_train_hp[:,0]=="Stricturing"), "1",y_2_train_hp[:,0])
y_2_train_hp[:,0] = np.where((y_2_train_hp[:,0]=="Not-stricturing"), "0" ,y_2_train_hp[:,0])
y_2_train_hp = np.core.records.fromarrays(y_2_train_hp.transpose(), names='col1, col2', formats = 'bool, f8')

y_3_train_hp[:,0] = np.where((y_3_train_hp[:,0]=="Stricturing"), "1",y_3_train_hp[:,0])
y_3_train_hp[:,0] = np.where((y_3_train_hp[:,0]=="Not-stricturing"), "0" ,y_3_train_hp[:,0])
y_3_train_hp = np.core.records.fromarrays(y_3_train_hp.transpose(), names='col1, col2', formats = 'bool, f8')

y_4_train_hp[:,0] = np.where((y_4_train_hp[:,0]=="Stricturing"), "1",y_4_train_hp[:,0])
y_4_train_hp[:,0] = np.where((y_4_train_hp[:,0]=="Not-stricturing"), "0" ,y_4_train_hp[:,0])
y_4_train_hp = np.core.records.fromarrays(y_4_train_hp.transpose(), names='col1, col2', formats = 'bool, f8')

y_5_train_hp[:,0] = np.where((y_5_train_hp[:,0]=="Stricturing"), "1",y_5_train_hp[:,0])
y_5_train_hp[:,0] = np.where((y_5_train_hp[:,0]=="Not-stricturing"), "0" ,y_5_train_hp[:,0])
y_5_train_hp = np.core.records.fromarrays(y_5_train_hp.transpose(), names='col1, col2', formats = 'bool, f8')

y_1_test[:,0] = np.where((y_1_test[:,0]=="Stricturing"), "1",y_1_test[:,0])
y_1_test[:,0] = np.where((y_1_test[:,0]=="Not-stricturing"), "0" ,y_1_test[:,0])
y_1_test = np.core.records.fromarrays(y_1_test.transpose(), names='col1, col2', formats = 'bool, f8')

y_2_test[:,0] = np.where((y_2_test[:,0]=="Stricturing"), "1",y_2_test[:,0])
y_2_test[:,0] = np.where((y_2_test[:,0]=="Not-stricturing"), "0" ,y_2_test[:,0])
y_2_test = np.core.records.fromarrays(y_2_test.transpose(), names='col1, col2', formats = 'bool, f8')

y_3_test[:,0] = np.where((y_3_test[:,0]=="Stricturing"), "1",y_3_test[:,0])
y_3_test[:,0] = np.where((y_3_test[:,0]=="Not-stricturing"), "0" ,y_3_test[:,0])
y_3_test = np.core.records.fromarrays(y_3_test.transpose(), names='col1, col2', formats = 'bool, f8')

y_4_test[:,0] = np.where((y_4_test[:,0]=="Stricturing"), "1",y_4_test[:,0])
y_4_test[:,0] = np.where((y_4_test[:,0]=="Not-stricturing"), "0" ,y_4_test[:,0])
y_4_test = np.core.records.fromarrays(y_4_test.transpose(), names='col1, col2', formats = 'bool, f8')

y_5_test[:,0] = np.where((y_5_test[:,0]=="Stricturing"), "1",y_5_test[:,0])
y_5_test[:,0] = np.where((y_5_test[:,0]=="Not-stricturing"), "0" ,y_5_test[:,0])
y_5_test = np.core.records.fromarrays(y_5_test.transpose(), names='col1, col2', formats = 'bool, f8')


#Grid Search Variables

params = dict()
params['min_samples_leaf'] = (1,2,3,4,5,6)
params['max_features'] = ('sqrt', 'log2', None)
params['min_samples_split'] = (2,3,4,5,6)
params['max_depth'] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30, None]
params['n_estimators'] = [100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

rfc = RandomSurvivalForest(bootstrap=True, oob_score=True,random_state=42)

# define evaluation
cv = KFold(n_splits=3, shuffle=True, random_state=42)
#cv = RepeatedStratifiedKFold(n_splits=7, n_repeats=3, random_state=1)
# define the search
search = BayesSearchCV(estimator=rfc, search_spaces=params, n_jobs=-1, cv=cv, verbose=1, n_iter=60)
# perform the search
result = search.fit(X_4_train_hp, y_4_train_hp)
# report the best result
print(search.best_score_)
print(search.best_params_)

# Get generalised CV

best_model_4 = result.best_estimator_
# evaluate model on the hold out dataset
#y_3_hat = best_model_3.predict(X_3_test)
# evaluate the model
cindex_4 = best_model_4.score(X_4_test, y_4_test)
# report progress
print(cindex_4)
#%% Random Survival Forest Fit

print('#################################\n Random Survival Forest \n#################################')

# Model Fitting

rsf = RandomSurvivalForest(n_estimators=1000,
                           min_samples_split=2,
                           min_samples_leaf=1,
                           max_features="sqrt",
                           max_depth=None,
                           n_jobs=-1,
                           random_state=42)
fit = rsf.fit(X_train, y_train)

#Concordance Index
train_cindex = rsf.score(X_train, y_train)

print('C-Index on training data: %.3f' %(train_cindex))

#y_pred = rsf.predict(X_test)

test_cindex = rsf.score(X_test, y_test)

print('C-Index on test data: %.3f' %(test_cindex))

#%% AUC over time

#AUC over time training

train_times = np.arange(1, 49, 1)
# estimate performance on training data, thus use `va_y` twice.
bob_auc, bob_mean_auc = cumulative_dynamic_auc(y_train, y_train, rsf.predict(X_train), train_times)

plt.plot(train_times, bob_auc, marker="o", color="r")
plt.axhline(bob_mean_auc, linestyle="--", color="r")
plt.xlabel("Years of Follow Up")
plt.ylabel("Time-dependent AUC")
plt.grid(True)

#AUC over time testing

test_times = np.arange(1, 54, 1)
# estimate performance on training data, thus use `va_y` twice.
bob_auc, bob_mean_auc = cumulative_dynamic_auc(y_train, y_test, rsf.predict(X_test), test_times)

plt.plot(test_times, bob_auc, marker="o", color="b")
plt.axhline(bob_mean_auc, linestyle="--", color="b")
plt.xlabel("Years of Follow Up")
plt.ylabel("Time-dependent AUC")
plt.legend(['Training Data', 'Avg. Training AUC', 'Testing Data', 'Avg. Testing AUC'], loc="lower left")
plt.grid(True)
    

#%% Feature Importances
import eli5
from eli5.sklearn import PermutationImportance
import webbrowser

howmany=len(sel_ge)
#Had to put feature names in manually for PCA 
#featnames = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10',
#             'PC11','PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19',
#             'PC20','PC21','PC22','PC23','PC24','PC25','PC26','PC27','PC28',
#             'PC29','PC30','PC31','PC32','PC33','PC34','PC35','PC36','PC37','PC38','PC39','PC40',
#             'PC41','PC42','PC43','PC44','PC45','PC46','PC47','PC48','PC49','PC50','PC51',
#             'PC52','PC53','PC54','PC55','PC56','PC57','PC58','PC59','PC60','PC61',
#             'PC62','PC63','PC64','PC65','PC66','PC67','PC68','PC69','PC70','PC71','PC72',	'PC73']
    
featnames = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10',
             'PC11','PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19',
             'PC20','PC21','PC22','PC23','PC24','PC25','PC26','PC27','PC28',
             'PC29','PC30','PC31','PC32','PC33','PC34']

many = len(featnames)

# n_iter is number of times you shuffle the data (keep )
perm = PermutationImportance(rsf, n_iter=15, random_state=42)
perm.fit(X_test, y_test)
importances=eli5.show_weights(perm, feature_names=sel_ge, top=howmany)
#importances=eli5.show_weights(perm, feature_names=featnames, top=many)
with open('C:/Users/iss1g18/OneDrive - University of Southampton/Documents/ML_IBD_2021/rsf/nodsig_test_feat.htm','wb') as f:   # Use some reasonable temp name
    f.write(importances.data.encode("UTF-8"))

# open an HTML file on my own (Windows) computer
url = r'C:/Users/iss1g18/OneDrive - University of Southampton/Documents/ML_IBD_2021/rsf/nodsig_test_feat.htm'
webbrowser.open(url,new=2)
