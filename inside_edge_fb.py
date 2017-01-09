#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:44:07 2016

@author: Jared
"""

#%%
# Import Packages
get_ipython().magic('matplotlib inline')

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

from sklearn import cross_validation
from sklearn.grid_search import RandomizedSearchCV
from sklearn import metrics as skm

from scipy.stats import uniform as sp_rand
from scipy.stats import randint as sp_randint

import xgboost as xgb
from sklearn.cross_validation import KFold

import sys
import operator
import random
import timeit

sys.path.append("../modules")

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import pylab as p
#%%
tic0=timeit.default_timer()
#%%
#tic=timeit.default_timer()
#xls = pd.ExcelFile('MLB_Challenge_Data updated 10-21-16.xlsx')
#toc = timeit.default_timer()
#
#dataset = xls.parse('MLB_Challenge_Data updated 10-2')
#
#print('Load Time',toc - tic)
##Make it into a csv for faster load times
##only need to do this step once
#dataset.to_csv('mlb_data.csv', index=False)

#%%
#NOTE MAY BE LEAK IN RECORD NUMBER
#%%
tic=timeit.default_timer()
#dataset_old = pd.read_csv('mlb_data.csv',header=0)
dataset = pd.read_csv('MLB_Challenge_Data 2015 upd 11-2-16.csv',header=0)
updates = pd.read_csv('OU2wB-T.csv',header=0)
toc=timeit.default_timer()
print('Load Time',toc - tic)
#%%
dataset.fillna(-1000,inplace=True)


#%%
dataset.rename(columns={'Actual_PTS':'points','RECORDNUM':'id'},inplace=True)
updates.rename(columns={'RECORDNUM':'id'},inplace=True)
#dataset.columns = [x.lower() for x in dataset.columns]
dataset.columns = [x.replace (" ", "_") for x in dataset.columns]
dataset.columns = [x.replace ("-", "_") for x in dataset.columns]
#%%
dataset.sort_values(by='id',inplace=True)
updates.sort_values(by='id',inplace=True)

dataset['roof'] = updates['roof2']
dataset['Current_O/U'] = updates['OU2']
dataset['elevation'] = updates['elevation2']
dataset['Hitter_Pitcher_B_T'] = updates['Hitter-Pitcher B-T 2']
#%%
object_cols = []
object_hash_cols = []
for feature in dataset.columns:
    if dataset[feature].dtype.name == 'object':
        object_cols.append(feature)
        feature_name_hash = feature + str('_hash')
        object_hash_cols.append(feature_name_hash)
        dataset[feature_name_hash] = pd.factorize(dataset[feature])[0]
#%%

is_sub_run = False
#is_sub_run = True
use_oof = False
#use_oof = True
random_seed = 5
if (is_sub_run):
    train = dataset.loc[dataset['points'] != -1000 ]
    test = dataset.loc[dataset['points'] == -1000 ]
else:
    if use_oof:
        train = dataset.loc[dataset['points'] != -1000 ]
        test = dataset.loc[dataset['points'] == -1000 ]
    else:
        train, test = cross_validation.train_test_split(dataset.loc[dataset['points'] != -1000], test_size = 0.3, random_state = random_seed)
    
#        train = dataset[(dataset['Week'] <= 19) & (dataset['points'] != -1000)].copy()
#        test = dataset[(dataset['Week'] > 19) & (dataset['points'] != -1000)].copy()


if use_oof:
    nfolds = 5
    #nfolds = 1
    if nfolds > 1:
        folds = KFold(len(train), n_folds = nfolds, shuffle = True, random_state = 111)
    else:
        folds = [(slice(None, None),slice(None,None))]
#%%
def get_mse(df,col = 'pred'):
    print('mse',np.sqrt(np.square(df[col] - df['points']).mean()))
#%%
#np.sqrt(np.square(test.points - test.points.mean()).mean())
#%%
#testing leakage
#if is_sub_run:
#    test['order_by_week'] = 1
#    test['order_by_week'] = test.groupby(['Week'])['order_by_week'].cumsum()
#%%
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()
#%%
import re
from io import BytesIO
_NODEPAT = re.compile(r'(\d+):\[(.+)\]')
_LEAFPAT = re.compile(r'(\d+):(leaf=.+)')
_EDGEPAT = re.compile(r'yes=(\d+),no=(\d+),missing=(\d+)')
_EDGEPAT2 = re.compile(r'yes=(\d+),no=(\d+)')
def _parse_node(graph, text):
    """parse dumped node"""
    match = _NODEPAT.match(text)
    if match is not None:
        node = match.group(1)
        graph.node(node, label=match.group(2), shape='circle')
        return node
    match = _LEAFPAT.match(text)
    if match is not None:
        node = match.group(1)
        graph.node(node, label=match.group(2), shape='box')
        return node
    raise ValueError('Unable to parse node: {0}'.format(text))


def _parse_edge(graph, node, text, yes_color='#0000FF', no_color='#FF0000'):
    """parse dumped edge"""
    try:
        match = _EDGEPAT.match(text)
        if match is not None:
            yes, no, missing = match.groups()
            if yes == missing:
                graph.edge(node, yes, label='yes', color=yes_color)
                graph.edge(node, no, label='no', color=no_color)
            else:
                graph.edge(node, yes, label='yes', color=yes_color)
                graph.edge(node, no, label='no', color=no_color)
            return
    except ValueError:
        pass
    match = _EDGEPAT2.match(text)
    if match is not None:
        yes, no = match.groups()
        graph.edge(node, yes, label='yes', color=yes_color)
        graph.edge(node, no, label='no', color=no_color)
        return
    raise ValueError('Unable to parse edge: {0}'.format(text))


def to_graphviz(booster, fmap='', num_trees=0, rankdir='UT',
                yes_color='#0000FF', no_color='#FF0000', **kwargs):

    """Convert specified tree to graphviz instance. IPython can automatically plot the
    returned graphiz instance. Otherwise, you should call .render() method
    of the returned graphiz instance.
    Parameters
    ----------
    booster : Booster, XGBModel
        Booster or XGBModel instance
    fmap: str (optional)
       The name of feature map file
    num_trees : int, default 0
        Specify the ordinal number of target tree
    rankdir : str, default "UT"
        Passed to graphiz via graph_attr
    yes_color : str, default '#0000FF'
        Edge color when meets the node condition.
    no_color : str, default '#FF0000'
        Edge color when doesn't meet the node condition.
    kwargs :
        Other keywords passed to graphviz graph_attr
    Returns
    -------
    ax : matplotlib Axes
    """

    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError('You must install graphviz to plot tree')

    if not isinstance(booster, (xgb.Booster, xgb.XGBModel)):
        raise ValueError('booster must be Booster or XGBModel instance')

    if isinstance(booster, xgb.XGBModel):
        booster = booster.booster()

    tree = booster.get_dump(fmap=fmap)[num_trees]
    tree = tree.split()

    kwargs = kwargs.copy()
    kwargs.update({'rankdir': rankdir})
    graph = Digraph(graph_attr=kwargs)

    for i, text in enumerate(tree):
        if text[0].isdigit():
            node = _parse_node(graph, text)
        else:
            if i == 0:
                # 1st string must be node
                raise ValueError('Unable to parse given string as tree')
            _parse_edge(graph, node, text, yes_color=yes_color,
                        no_color=no_color)

    return graph


def plot_tree(booster, fmap='', num_trees=0, rankdir='UT', ax=None, **kwargs):
    """Plot specified tree.
    Parameters
    ----------
    booster : Booster, XGBModel
        Booster or XGBModel instance
    fmap: str (optional)
       The name of feature map file
    num_trees : int, default 0
        Specify the ordinal number of target tree
    rankdir : str, default "UT"
        Passed to graphiz via graph_attr
    ax : matplotlib Axes, default None
        Target axes instance. If None, new figure and axes will be created.
    kwargs :
        Other keywords passed to to_graphviz
    Returns
    -------
    ax : matplotlib Axes
    """

    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as image
    except ImportError:
        raise ImportError('You must install matplotlib to plot tree')

    if ax is None:
        _, ax = plt.subplots(1, 1)

    g = to_graphviz(booster, fmap=fmap, num_trees=num_trees, rankdir=rankdir, **kwargs)

    s = BytesIO()
    s.write(g.pipe(format='png'))
    s.seek(0)
    img = image.imread(s)

    ax.imshow(img)
    ax.axis('off')
    return ax
#%%
def fit_xgb_model(train, test, params, xgb_features, num_rounds = 10, num_rounds_es = 200000,
                  use_early_stopping = True, print_feature_imp = False,
                  random_seed = 123, calculate_rmse = True, save_model = False
                  ):

    tic=timeit.default_timer()
    random.seed(random_seed)
    np.random.seed(random_seed)

    X_train, X_watch = cross_validation.train_test_split(train, test_size=0.2,random_state=random_seed)
    train_data = X_train[xgb_features].values
    train_data_full = train[xgb_features].values
    watch_data = X_watch[xgb_features].values
    train_points = X_train['points'].astype(float).values
    train_points_full = train['points'].astype(float).values
    watch_points = X_watch['points'].astype(float).values
    test_data = test[xgb_features].values


    dtrain = xgb.DMatrix(train_data, train_points)
    dtrain_full = xgb.DMatrix(train_data_full, train_points_full)

    dwatch = xgb.DMatrix(watch_data, watch_points)
    dtest = xgb.DMatrix(test_data)
    watchlist = [(dtrain, 'train'),(dwatch, 'watch')]

    if use_early_stopping:
        xgb_classifier = xgb.train(params, dtrain, num_boost_round=num_rounds_es, evals=watchlist,
                            early_stopping_rounds=100, verbose_eval=50)
        y_pred = xgb_classifier.predict(dtest,ntree_limit=xgb_classifier.best_iteration)
    else:
        xgb_classifier = xgb.train(params, dtrain_full, num_boost_round=num_rounds, evals=[(dtrain_full,'train')],
                            verbose_eval=50)
        y_pred = xgb_classifier.predict(dtest)


    if(print_feature_imp):
        create_feature_map(xgb_features)
        imp_dict = xgb_classifier.get_fscore(fmap='xgb.fmap')
        imp_dict = sorted(imp_dict.items(), key=operator.itemgetter(1),reverse=True)
        print('{0:<20} {1:>5}'.format('Feature','Imp'))
        print("--------------------------------------")
        num_to_print = 40
        num_printed = 0
        for i in imp_dict:
            num_printed = num_printed + 1
            if (num_printed > num_to_print):
                continue
            print ('{0:20} {1:5.0f}'.format(i[0], i[1]))
    columns = ['pred']

    result_xgb_df = pd.DataFrame(index=test.id, columns=columns,data=y_pred)
    if(is_sub_run):
        print('is a submission run')
        result_xgb_df.reset_index('id',inplace=True)
        result_xgb_df = pd.merge(result_xgb_df,test[['id','points'] + xgb_features],left_on = ['id'],
                                   right_on = ['id'],how='left')
    else:
        if(calculate_rmse):
            result_xgb_df.reset_index('id',inplace=True)
            result_xgb_df = pd.merge(result_xgb_df,test[['id','points'] + xgb_features],left_on = ['id'],
                                   right_on = ['id'],how='left')

            result_xgb_df['error_sq'] = np.square((result_xgb_df['pred'] - result_xgb_df['points']))
            result_xgb_df['error'] = result_xgb_df['points'] - result_xgb_df['pred']
            #reorder columns for convenience
            cols = result_xgb_df.columns.tolist()
            cols.remove('id')
            cols.remove('points')
            cols.remove('pred')
            cols.remove('error_sq')
            cols.remove('error')
            result_xgb_df = result_xgb_df[['id','points','pred','error_sq','error'] + cols]
            print('rmse',round(np.sqrt(result_xgb_df['error_sq'].mean()),5))
    toc=timeit.default_timer()
    print('xgb Time',toc - tic)
    if not save_model:
        return result_xgb_df
    else:
        return (result_xgb_df,xgb_classifier)

#%%


#base_features = []
#for column in dataset.columns:
#    base_features.append(column)
#%%
base_features = ['Week','HitterStatus','Hitter_Pos','PitcherSOTend','PitcherSide',
 'HitterGLYCategory','PitcherPcntlGroup','HitterPcntlGroup','PitcherGLYCategory',
 'PitcherGB_FLY','HitterHomeAway','PA_H2H','H_H2H','AB_H2H','BAVG_H2H','SLG_H2H',
 'HR_H2H','wOBA_H2H','SO_H2H','WHAvg_H2H','PA_PGLY','H_PGLY','AB_PGLY','BAVG_PGLY',
 'SLG_PGLY','HR_PGLY','wOBA_PGLY','SO_PGLY','WHAvg_PGLY','PA_PTier','H_PTier',
 'AB_PTier','BAVG_PTier','SLG_PTier','HR_PTier','wOBA_PTier','SO_PTier','WHAvg_PTier',
 'PA_HGLY','H_HGLY','AB_HGLY','BAVG_HGLY','SLG_HGLY','wOBA_HGLY','WHAvg_HGLY',
 'PA_Last_5','H_Last_5','AB_Last_5','BAVG_Last_5','XBH_Last_5','wOBA_Last_5',
 'WHAvg_Last_5','PA_HomeAway','wOBA_HomeAway','WHAvg_HomeAway','PA_HomeAway___Pitcher',
 'WOBA_HomeAway_Pitcher','WHAvg_HomeAway___Pitcher','PA_vs_SO_Pitcher','H_SO_Pitcher',
 'AB_SO_Pitcher','BAVG_SO_Pitcher','SLG_SO_Pitcher','wOBA_vs_SO_Pitcher',
 'WHAvg_vs_SO_Pitcher','K%_vs_SO_Pitcher','PA_vs_GB_Pitcher','H_GB_Pitcher',
 'AB_GB_Pitcher','BAVG_GB_Pitcher','SLG_GB_Pitcher','wOBA_vs_GB_Pitcher','WHAvg_vs_GB_Pitcher',
 'H2H_MU_SCORE','PGLY_MU_SCORE','PTier_MU_SCORE','HGLY_MU_SCORE','SO_Tend_MU_SCORE','GB_Tend_MU_SCORE',
 'Last5_MU_SCORE','HomeAway_MU_SCORE','HomeAway_MU_SCORE___Pitcher','PlayToday_Likelihood',
 'MatchupScore3','Normalized_MU3','PA_vs_SP_Side','Park_Adj','Expected_wOBA','Fanduel_Pos',
 'Hitter_WHRating_Avg_Last15','Hitter_SLG_Last15','PA_Last15Days','Opp._Pitcher_WHRating_Avg',
 'Opp._Pitcher_SLG','Current_O/U','Hitter_Pitcher_B_T','Hitter_WHRating_Last5Days',
 'Hitter_WHRating_Last365','Pitcher_Good_Greats_Last_4','humidity','pressure',
 'condition','temp','chanceofrain','outtoleftimpact','outtorightimpact','DayNight',
 'elevation','roof','BOP','HitterStatus_hash',
 'Hitter_Pos_hash','PitcherSOTend_hash','PitcherSide_hash',
 'HitterGLYCategory_hash','PitcherPcntlGroup_hash','HitterPcntlGroup_hash',
 'PitcherGLYCategory_hash','PitcherGB_FLY_hash','HitterHomeAway_hash','K%_vs_SO_Pitcher_hash',
 'PlayToday_Likelihood_hash','Fanduel_Pos_hash','Hitter_Pitcher_B_T_hash',
 'condition_hash','DayNight_hash','roof_hash']
base_features = [feature for feature in base_features if feature not in object_cols]
#%%

#%%
#corr_mat = train.corr()


gen_info = ['Week','DayNight_hash','HitterStatus_hash','PlayToday_Likelihood_hash',
            'Hitter_Pos_hash','Fanduel_Pos_hash','BOP','HitterHomeAway_hash',
            'PitcherSide_hash','Hitter_Pitcher_B_T_hash']
ie_info = ['PitcherSOTend_hash','HitterGLYCategory_hash','PitcherPcntlGroup_hash',
           'HitterPcntlGroup_hash','PitcherGLYCategory_hash','PitcherGB_FLY_hash']
h2h_info = [ 'PA_H2H','H_H2H','AB_H2H','BAVG_H2H','SLG_H2H','HR_H2H','wOBA_H2H',
            'SO_H2H','WHAvg_H2H']
pgly_info = [ 'PA_PGLY','H_PGLY','AB_PGLY','BAVG_PGLY','SLG_PGLY',
             'HR_PGLY','wOBA_PGLY','SO_PGLY','WHAvg_PGLY']
ptier_info = [ 'PA_PTier','H_PTier','AB_PTier','BAVG_PTier','SLG_PTier',
              'HR_PTier','wOBA_PTier','SO_PTier','WHAvg_PTier']
hgly_info = [ 'PA_HGLY','H_HGLY','AB_HGLY','BAVG_HGLY','SLG_HGLY',
             'wOBA_HGLY','WHAvg_HGLY']
last5_info = [ 'PA_Last_5','H_Last_5','AB_Last_5','BAVG_Last_5','XBH_Last_5',
              'wOBA_Last_5','WHAvg_Last_5']
homeaway_info = [ 'PA_HomeAway','wOBA_HomeAway','WHAvg_HomeAway','PA_HomeAway___Pitcher',
                 'WOBA_HomeAway_Pitcher','WHAvg_HomeAway___Pitcher']
pitcher_info = ['PA_vs_SO_Pitcher','H_SO_Pitcher','AB_SO_Pitcher','BAVG_SO_Pitcher',
 'SLG_SO_Pitcher','wOBA_vs_SO_Pitcher','WHAvg_vs_SO_Pitcher','PA_vs_GB_Pitcher',
 'H_GB_Pitcher','AB_GB_Pitcher','BAVG_GB_Pitcher','SLG_GB_Pitcher','wOBA_vs_GB_Pitcher',
 'WHAvg_vs_GB_Pitcher']      
matchup_info = [ 'H2H_MU_SCORE','PGLY_MU_SCORE','PTier_MU_SCORE','HGLY_MU_SCORE',
 'SO_Tend_MU_SCORE','GB_Tend_MU_SCORE','Last5_MU_SCORE','HomeAway_MU_SCORE',
 'HomeAway_MU_SCORE___Pitcher','MatchupScore3','Normalized_MU3']
hitter_history_info = ['PA_vs_SP_Side', 'Expected_wOBA','Hitter_WHRating_Avg_Last15',
 'Hitter_SLG_Last15', 'PA_Last15Days', 'Hitter_WHRating_Last5Days','Hitter_WHRating_Last365']
pitcher_history_info = ['Pitcher_Good_Greats_Last_4', 'Opp._Pitcher_WHRating_Avg',
 'Opp._Pitcher_SLG']
weather_info = [ 'humidity',
 'pressure',
 'temp',
 'chanceofrain',
 'outtoleftimpact',
 'outtorightimpact',
 'condition_hash']
ballpark_info = ['elevation','roof_hash','Park_Adj']
vegas_info = ['Current_O/U']
#%%
xgb_features = []
xgb_features += base_features
#xgb_features += features_by_med

params = {'learning_rate': 0.005,
              'subsample': 0.99,
              'reg_alpha': 0.3,
#              'lambda': 0.995,
              'gamma': 0.1,
              'seed': 5,
              'colsample_bytree': 0.3,
#              'n_estimators': 100,
              'objective': 'reg:linear',
              'eval_metric':'rmse',
#              'min_child_weight': 2,
              'max_depth': 7,
#              'max_depth': 3,
              }

#params = {'learning_rate': 0.005,
#              'subsample': 0.6,
#              'reg_alpha': 0.1,
##              'lambda': 0.995,
#              'gamma': 0.1,
#              'seed': 5,
#              'colsample_bytree': 0.3,
##              'n_estimators': 100,
#              'objective': 'reg:linear',
#              'eval_metric':'rmse',
##              'min_child_weight': 2,
#              'max_depth': 1,
##              'max_depth': 3,
#              }
              
              
#xgb_features.remove('Week')
#xgb_features.remove('BOP')

#this is used to find early stopping rounds, then scale up when using all of dataset
#as number of rounds should be proportional to size of dataset

#xgb_features = [x for x in xgb_features if x not in vegas_info]

#xgb_features = gen_info + weather_info + vegas_info + last5_info + pgly_info + hgly_info + hitter_history_info
(result_xgb_1,model_1) = fit_xgb_model(train,test,params, xgb_features,use_early_stopping = True,
                              print_feature_imp = True, random_seed = 6,save_model=True)



#result_xgb_1 = fit_xgb_model(train,test,params, xgb_features,use_early_stopping = True,
#                              print_feature_imp = True, random_seed = 6)
#num_rounds_1 = 1268
#if is_sub_run:
#    num_rounds_1 /= (0.8 * 0.7)
#else:
#    num_rounds_1 /= (0.8)
#num_rounds_1 = int(num_rounds_1)
#result_xgb_1 = fit_xgb_model(train,test,params,xgb_features,
#                              num_rounds = num_rounds_1, print_feature_imp = True,
#                              use_early_stopping = False,random_seed = 6)


result_xgb_1.index = result_xgb_1.id

DF_LIST = []
for bop in range(1,10):
    result_xgb_temp = fit_xgb_model(train[train['BOP'] == bop],test[test['BOP'] == bop],params, xgb_features,use_early_stopping = True,
                                  print_feature_imp = True, random_seed = 6)
    DF_LIST.append(result_xgb_temp)

result_xgb_2 = pd.concat(DF_LIST,ignore_index=True)
result_xgb_2.index = result_xgb_2['id']

if not is_sub_run:
    print('rmse',round(np.sqrt(result_xgb_1['error_sq'].mean()),5))
    print('just mean rmse',np.sqrt(np.square(test.points - train.points.mean()).mean()))

if use_oof:
    DF_LIST_1 = []
    for (inTr, inTe) in folds:
        xtr = train.iloc[inTr].copy()
        xte = train.iloc[inTe].copy()
    
        num_rounds_1 = 1268
        if is_sub_run:
            num_rounds_1 /= (0.8)
        else:
            num_rounds_1 /= (0.8)
        num_rounds_1 = int(num_rounds_1)
        result_xgb_temp = fit_xgb_model(xtr,xte,params,xgb_features,
                                      num_rounds = num_rounds_1, print_feature_imp = True,
                                      use_early_stopping = False,random_seed = 6)
        DF_LIST_1.append(result_xgb_temp)
    res_oob_xgb_1 = pd.concat(DF_LIST_1,ignore_index=True)
    res_oob_xgb_1.index = res_oob_xgb_1['id']
    
    result_xgb_1 = res_oob_xgb_1.copy()
    del DF_LIST_1

#%%
result_xgb_ens = result_xgb_1.copy()
a1 = 1
a2 = 1
result_xgb_ens['pred'] = (result_xgb_1['pred'] * a1 + result_xgb_2['pred'] * a2) / (a1 + a2)

if not is_sub_run:
    result_xgb_ens['error_sq'] = np.square((result_xgb_ens['pred'] - result_xgb_ens['points']))
    print('rmse',round(np.sqrt(result_xgb_ens['error_sq'].mean()),5))
    print('just mean rmse',np.sqrt(np.square(test.points - train.points.mean()).mean()))
#%%
#temp = to_graphviz(model_1,fmap='xgb.fmap',rankdir = 'LR')
#temp.save('simplified_example')
#temp.render('simplified_example')
#%%

#importance = model_1.get_fscore(fmap='xgb.fmap')
importance = model_1.get_score(fmap='xgb.fmap',importance_type='gain')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

df = df[df.index >= 90]
plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('Feature Importance')
plt.gcf().savefig('feature_importance_xgb_all.png',bbox_inches='tight', dpi=500)
#%%

#p.figure()
#plt.hist2d(result_xgb_1.points, result_xgb_1.pred, bins=[40,40],range=[(0,40),(0,40)],cmin=1)
#plt.colorbar()
#plt.tick_params(axis='both', which='major', labelsize=15)
#plt.tick_params(axis='both', which='major', labelsize=20)
#plt.gcf().set_size_inches(8, 4)
#plt.rcParams.update({'font.size': 22})
##plt.title('Holdout Predictions')
#plt.xlabel('Actual Points')
#plt.ylabel('Predicted Points')
#p.savefig('Images/pred_vs_actual_hist.png', bbox_inches='tight', dpi=500)
#
#p.figure()
#plt.scatter(result_xgb_1.points, result_xgb_1.error,alpha=0.25)
#plt.xlim(-0.5, 71)
#plt.ylim(-20, 60)
#plt.tick_params(axis='both', which='major', labelsize=15)
#plt.tick_params(axis='both', which='major', labelsize=20)
#plt.gcf().set_size_inches(8, 4)
#plt.rcParams.update({'font.size': 22})
#plt.grid()
#plt.xlabel('Actual Points')
#plt.ylabel('Error (Actual - Pred)')
#p.savefig('Images/error_vs_actual_scatter.png', bbox_inches='tight', dpi=500)

#%%
#if is_sub_run:
#    res_oob_xgb_merged = pd.merge(train[['id','points']],res_oob_xgb_1,left_on = ['id'],
#                       right_on = ['id'],how='left')
#    get_mse(res_oob_xgb_merged)
#    print('pred mean mse',np.sqrt(np.square(train.points - train.points.mean()).mean()))

#result_xgb_samp_1 = result_xgb_1.sample(frac = 0.1,random_state = 3)
#%%
hit_pos_dict = pd.Series(train.Hitter_Pos.values,index=train.Hitter_Pos_hash).to_dict()
hit_pitch_b_t_dict = pd.Series(train.Hitter_Pitcher_B_T.values,index=train.Hitter_Pitcher_B_T_hash).to_dict()
#%%
bins_bavg = [-1,0,0.1,0.2,0.3,0.4,0.5,1.0]
train['BAVG_PGLY_binned'] = np.digitize(train['BAVG_PGLY'],bins_bavg,right=True)
bins_last5_mu = [-1,10,20,30,40,50,60,70,80,90,100]
train['Last5_MU_SCORE_binned'] = np.digitize(train['Last5_MU_SCORE'],bins_last5_mu,right=True)
#%%

def make_graph(dataset,col_name,title_name,xlabel='x',ylabel='Mean Points',
               use_custom_dict = False, custom_dict = {},rotation='horizontal',
               save_pic = True, pic_name = 'Images/temp.png'):
    p.figure()
    plt.tick_params(axis='both', which='major', labelsize=15)
    p.legend(prop={'size':20})
    plt.grid()
#    ax = plt.gca()
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.gcf().set_size_inches(8, 4)
    plt.rcParams.update({'font.size': 22})
    gr = dataset.groupby(col_name)['points'].mean()
    gr_errors = dataset.groupby(col_name)['points'].sem()
    
    plt.errorbar(gr.index,gr,gr_errors,fmt='o')
    plt.title(title_name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if use_custom_dict:
        my_ticks = [custom_dict[x] for x in gr.index.values]
        plt.xticks(gr.index, my_ticks,rotation=rotation)
#    plt.tight_layout()
    xticks, xticklabels = plt.xticks()
    # shift half a step to the left
    # x0 - (x1 - x0) / 2 = (3 * x0 - x1) / 2
    xmin = (3*xticks[0] - xticks[1])/2
    # shaft half a step to the right
    xmax = (3*xticks[-1] - xticks[-2])/2
    plt.xlim(xmin, xmax)
    plt.xticks(xticks)
    if save_pic:
        p.savefig(pic_name, bbox_inches='tight', dpi=500)



#make_graph(train,'Hitter_Pos_hash','Mean Points vs Hitter Position',xlabel='Hitter Position',
#           use_custom_dict = True, custom_dict = hit_pos_dict,pic_name = 'Images/Hitter_Pos_hash.png')    
#make_graph(train,'Hitter_Pitcher_B_T_hash','Mean Points vs Hitter Pitcher B_T',xlabel='Hitter Pitcher B_T',
#           use_custom_dict = True, custom_dict = hit_pitch_b_t_dict,rotation='vertical',pic_name = 'Images/Hitter_Pitcher_B_T.png')    
#make_graph(train,'BOP','Mean Points vs Batting Order',xlabel='BOP',pic_name = 'Images/BOP.png')
#make_graph(train,'Week','Mean Points vs Week',xlabel='Week',pic_name = 'Images/Week.png')
#make_graph(train,'Park_Adj','Mean Points vs Park Adjustment',xlabel='Park Adjustment',pic_name = 'Images/Park_Adj.png')
#make_graph(train,'BAVG_PGLY_binned','Mean Points vs BAVG PGLY',xlabel='BAVG PGLY',
#           use_custom_dict = True, custom_dict = bins_bavg,pic_name = 'Images/BAVG_PGLY.png')
#make_graph(train,'Last5_MU_SCORE_binned','Mean Points vs Last5 Mu Score',xlabel='Last5 Mu Score',
#           use_custom_dict = True, custom_dict = bins_last5_mu,pic_name = 'Images/Last5_Mu_Score.png')
#make_graph(train,'Current_O/U','Mean Points vs O/U',xlabel='Over / Under',pic_name = 'Images/CurrentOverUnder.png')

#%%
#if is_sub_run:
#    result_xgb_1 = pd.merge(result_xgb_1,test[['id','order_by_week','outtorightimpact']],left_on = ['id'],
#                           right_on = ['id'],how='left')
#    corr_test = result_xgb_1.corr()
#%%

if is_sub_run:
    submission = result_xgb_1[['id','pred']].copy()
    submission.rename(columns={'id':'RECORDNUM','pred':'Predicted Points'},inplace=True)
    submission.sort_values(by='RECORDNUM')
    submission.to_csv('sports_go_sports_submission.csv',index=False)
#%%
toc = timeit.default_timer()
print('Total Time',toc - tic0)



