from sklearn.metrics import make_scorer, confusion_matrix
from math import sqrt
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from xgboost import XGBClassifier as XGB
from lightgbm import LGBMClassifier as LGBC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.neural_network import MLPClassifier as MLP
# from ray import tune

weight = 2.81640625
def MCC_scorer(clf, X, y, ):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1],
    MCC = (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return MCC
def Sn_scorer(clf, X, y, ):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1],
    Sn = TP /(TP + FN)
    return Sn
def Sp_scorer(clf, X, y, ):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    Sp = TN / (TN + FP)
    return Sp
def all_scores(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    MCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    Sn = TP / (TP + FN)
    Sp = TN / (TN + FP)
    return MCC, Sn, Sp



def test_svm(best_trial, X_train, X_test, y_train, y_test ):
    kernel = best_trial['kernel'] #‘linear’, ‘poly’, ‘rbf’
    C = best_trial['C']
    gamma = best_trial['gamma']
    clf = SVC(kernel=kernel, C=C, gamma=gamma, class_weight='balanced',probability=True)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    y_test_probas = clf.predict_proba(X_test)
    y_test_probas = y_test_probas[:, 1]
    acc = metrics.accuracy_score(y_test,y_test_pred)
    ba = metrics.balanced_accuracy_score(y_test, y_test_pred)
    f1 = metrics.f1_score(y_test, y_test_pred)
    auc = metrics.roc_auc_score(y_test, y_test_probas)
    mcc, sn, sp = all_scores(y_test,y_test_pred)
    scores = {'test_acc':acc, 'test_ba':ba, 'test_f1':f1, 'test_auc':auc,
              'test_mcc':mcc, 'test_sn':sn, 'test_sp':sp}
    print(scores)
    return scores


def test_RF(best_trial, X_train, X_test, y_train, y_test):
    n_estimators = int(best_trial['n_estimators'])
    max_features = best_trial['max_features']
    max_depth = int(best_trial['max_depth'])
    min_samples_split = int(best_trial['min_samples_split'])
    min_samples_leaf = int(best_trial['min_samples_leaf'])
    max_leaf_nodes = best_trial['max_leaf_nodes']
    clf = RFC(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth,
              min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,
              class_weight='balanced', random_state=1)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    y_test_probas = clf.predict_proba(X_test)
    y_test_probas = y_test_probas[:,1]
    acc = metrics.accuracy_score(y_test,y_test_pred)
    ba = metrics.balanced_accuracy_score(y_test, y_test_pred)
    f1 = metrics.f1_score(y_test, y_test_pred)
    auc = metrics.roc_auc_score(y_test, y_test_probas)
    mcc, sn, sp = all_scores(y_test,y_test_pred)
    scores = {'test_acc':acc, 'test_ba':ba, 'test_f1':f1, 'test_auc':auc,
              'test_mcc':mcc, 'test_sn':sn, 'test_sp':sp}
    print(scores)
    return scores


def test_xgb(best_trial, X_train, X_test, y_train, y_test):
    n_estimators = int(best_trial['n_estimators'])
    eta = best_trial['eta']
    min_child_weight = best_trial['min_child_weight']
    max_depth = int(best_trial['max_depth'])
    gamma = best_trial['gamma']
    subsample = best_trial['subsample']
    colsample_bytree = best_trial['colsample_bytree']
    reg_lambda = best_trial['reg_lambda']
    alpha = best_trial['alpha']
    clf = XGB(n_estimators=n_estimators, eta=eta, min_child_weight=min_child_weight, max_depth=max_depth,
              gamma=gamma, subsample=subsample,colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, alpha=alpha,
              scale_pos_weight=weight, eval_metric='auc', booster='gbtree', objective='binary:logistic',
              seed=1, tree_method='exact', n_jobs=5, random_state=1)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    y_test_probas = clf.predict_proba(X_test)
    y_test_probas = y_test_probas[:, 1]
    acc = metrics.accuracy_score(y_test, y_test_pred)
    ba = metrics.balanced_accuracy_score(y_test, y_test_pred)
    f1 = metrics.f1_score(y_test, y_test_pred)
    auc = metrics.roc_auc_score(y_test, y_test_probas)
    mcc, sn, sp = all_scores(y_test,y_test_pred)
    scores = {'test_acc':acc, 'test_ba':ba, 'test_f1':f1, 'test_auc':auc,
              'test_mcc':mcc, 'test_sn':sn, 'test_sp':sp}
    print(scores)
    return scores

def test_lgbm(best_trial, X_train, X_test, y_train, y_test):
    n_estimators = int(best_trial['n_estimators'])
    learning_rate = best_trial['learning_rate']
    max_depth = int(best_trial['max_depth'])
    num_leaves = 2 ** max_depth - 1
    subsample = best_trial['subsample']
    colsample_bytree = best_trial['colsample_bytree']
    reg_lambda = best_trial['reg_lambda']
    # alpha = best_trial['alpha']
    clf = LGBC(n_estimators=n_estimators, learning_rate=learning_rate,
               max_depth=max_depth, num_leaves=num_leaves,
               subsample=subsample, colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, 
               scale_pos_weight=weight, objective='binary', boosting_type='gbdt', n_jobs=1, random_state=1) #alpha=alpha,
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    y_test_probas = clf.predict_proba(X_test)
    y_test_probas = y_test_probas[:, 1]
    acc = metrics.accuracy_score(y_test, y_test_pred)
    ba = metrics.balanced_accuracy_score(y_test, y_test_pred)
    f1 = metrics.f1_score(y_test, y_test_pred)
    auc = metrics.roc_auc_score(y_test, y_test_probas)
    mcc, sn, sp = all_scores(y_test,y_test_pred)
    scores = {'test_acc':acc, 'test_ba':ba, 'test_f1':f1, 'test_auc':auc,
              'test_mcc':mcc, 'test_sn':sn, 'test_sp':sp}
    print(scores)
    return scores

def test_knn(best_trial, X_train, X_test, y_train, y_test):
    n_neighbors = int(best_trial['n_neighbors'])
    weights = best_trial['weights']
    algorithm =best_trial['algorithm']
    leaf_size = best_trial['leaf_size']

    clf = KNN(n_neighbors=n_neighbors, weights=weights, leaf_size=leaf_size, algorithm=algorithm,
              n_jobs=5, )
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    y_test_probas = clf.predict_proba(X_test)
    y_test_probas = y_test_probas[:, 1]
    acc = metrics.accuracy_score(y_test,y_test_pred)
    ba = metrics.balanced_accuracy_score(y_test, y_test_pred)
    f1 = metrics.f1_score(y_test, y_test_pred)
    auc = metrics.roc_auc_score(y_test, y_test_probas)
    mcc, sn, sp = all_scores(y_test,y_test_pred)
    scores = {'test_acc':acc, 'test_ba':ba, 'test_f1':f1, 'test_auc':auc,
              'test_mcc':mcc, 'test_sn':sn, 'test_sp':sp}
    print(scores)
    return scores

import pickle
def test_lr(best_trial, X_train, X_test, y_train, y_test):
    C = best_trial['C']
    clf = LR(C=C, penalty='l1',solver='liblinear',class_weight='balanced',random_state=1)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    y_test_probas = clf.predict_proba(X_test)
    y_test_probas = y_test_probas[:, 1]
    acc = metrics.accuracy_score(y_test,y_test_pred)
    ba = metrics.balanced_accuracy_score(y_test, y_test_pred)
    f1 = metrics.f1_score(y_test, y_test_pred)
    auc = metrics.roc_auc_score(y_test, y_test_probas)
    mcc, sn, sp = all_scores(y_test,y_test_pred)
    scores = {'test_acc':acc, 'test_ba':ba, 'test_f1':f1, 'test_auc':auc,
              'test_mcc':mcc, 'test_sn':sn, 'test_sp':sp}
    print(scores)
    return scores

def final_test_lr(best_trial,clf_str, X_train, X_test, y_train, y_test):
    C = best_trial['C']
    clf = LR(C=C, penalty='l1',solver='liblinear',class_weight='balanced',random_state=1)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    y_test_probas = clf.predict_proba(X_test)
    y_test_probas = y_test_probas[:, 1]
    acc = metrics.accuracy_score(y_test,y_test_pred)
    ba = metrics.balanced_accuracy_score(y_test, y_test_pred)
    f1 = metrics.f1_score(y_test, y_test_pred)
    auc = metrics.roc_auc_score(y_test, y_test_probas)
    mcc, sn, sp = all_scores(y_test,y_test_pred)
    scores = {'test_acc':acc, 'test_ba':ba, 'test_f1':f1, 'test_auc':auc,
              'test_mcc':mcc, 'test_sn':sn, 'test_sp':sp}
    print(scores)
    with open('models/'+clf_str+'.pkl', 'wb') as f:
        pickle.dump(clf, f)
    return scores