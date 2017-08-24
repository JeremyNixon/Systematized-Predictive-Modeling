import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import math
import json
import time
import random
import datetime as dt
from datetime import timedelta
import itertools

from sklearn import ensemble
from sklearn import linear_model
from sklearn import svm
from sklearn.linear_model import RandomizedLasso
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import sklearn.cross_validation
from sklearn import discriminant_analysis
from sklearn import naive_bayes
from sklearn import tree
from sklearn import cluster
from sklearn import metrics
from sklearn import cross_validation
from sklearn import svm

import xgboost as xgb

class apm():
    def __init__(self, x, y, task, test=None, hyperparameters='individual', debug=False):
        self.time = time.time()
        self.task = task
        self.hyperparameters = hyperparameters
        self.x = np.array(x)
        self.y = y
        if self.task == 'classification':
            if type(self.y[0]) != int:
                self.y, self.mapping = self.categorize(self.y)
            self.y = np.array(self.y, dtype = int)
        elif self.task == 'regression':
            self.y = np.array(self.y, dtype = float)
            
        if test is None:
            self.test = None
        else:
            self.test = self.feature_transform(np.array(test))
            
        # self.x = self.feature_transform(self.x)
        
        
        self.x_train, self.x_test, self.y_train, self.y_test = sklearn.cross_validation.train_test_split\
            (self.x, self.y, test_size = .20, random_state=42)
        self.x_train1 = self.x_train[:len(self.x_train)/2]
        self.x_train2 = self.x_train[len(self.x_train)/2:]
        self.y_train1 = self.y_train[:len(self.x_train)/2]
        self.y_train2 = self.y_train[len(self.x_train)/2:]
        self.stacking_train = self.x_train2
        self.stacking_test = self.x_test
        
        self.x1 = self.x[:len(self.x)/2]
        self.x2 = self.x[len(self.x)/2:]
        self.y1 = self.y[:len(self.x)/2]
        self.y2 = self.y[len(self.x)/2:]
        self.test_stacking_train = self.x2
        self.test_stacking_test = self.test

        self.debug = False
        self.stacking = False
        self.testing = False
        self.optimal_iterator = None
        
        if self.task == 'classification':
            if self.debug:
                self.algorithms = [self.logistic_regression]
                self.algorithm_names = ['logistic_predictions']
            else:
                self.algorithms = [self.xgboost, self.gradient_booster, self.random_forest,\
                      self.extremely_random_forest, self.logistic_regression, self.lda, self.qda]
                self.algorithm_names = ['xgboost_predictions','booster_predictions', 'forest_predictions', 'erf_predictions',\
                          'logistic_predictions', 'lda_predictions', 'qda_predictions', \
                           'xgboost_on_stack', 'booster_on_stack','forest_on_stack', \
                            'erf_on_stack', 'logistic_on_stack', 'lda_on_stack', 'qda_on_stack']
        elif self.task == 'regression':
            if self.debug:
                self.algorithms = [self.linear_regression, self.ridge_regression, self.lasso_regression,\
                                   self.gradient_booster, self.random_forest, self.extremely_random_forest]
                self.algorithm_names = ['linear_predictions', 'ridge_predictions', 'lasso_predictions',\
                                        'booster_predictions', 'forest_predictions', 'erf_predictions', 'linear_on_stack',\
                                        'ridge_on_stack', 'lasso_on_stack', 'booster_on_stack', 'forest_on_stack', 'erf_on_stack']
            else:
                self.algorithms = [self.linear_regression, self.ridge_regression, self.lasso_regression,\
                                   self.gradient_booster, self.random_forest, self.extremely_random_forest]
                self.algorithm_names = ['linear_predictions', 'ridge_predictions', 'lasso_predictions',\
                                        'booster_predictions', 'forest_predictions', 'erf_predictions', 'linear_on_stack',\
                                        'ridge_on_stack', 'lasso_on_stack', 'booster_on_stack', 'forest_on_stack', 'erf_on_stack']
                
        self.xgb_max_depth = None
        self.gb_n_estimators = None
        self.gb_max_depth = None
        self.rf_n_estimators = 400
        self.rf_max_features = "auto"
        self.erf_n_estimators = None
        self.erf_max_features = None
        self.log_reg_penalty = None
        self.log_reg_C = None
        self.qda_reg_param = None
        self.svc_C = None
        self.svc_kernel = None
        
        self.training_df = None
        self.testing_df = None        
        
    
    def accuracy(self, predictions):
        if len(predictions) == len(self.y_test):
            y_test = self.y_test
        elif len(predictions) == len(self.y_train2):
            y_test = self.y_train2
        elif len(predictions) == len(self.y2):
            y_test = self.y2
        else:
            print "Accuracy called with incorrectly sized predictions"
        if self.task == 'classification':
            count = 0
            for i in xrange(len(predictions)):
                if predictions[i] == y_test[i]:
                    count += 1
            accuracy = count/float(len(predictions))
        elif self.task == 'regression':
            error = [np.abs(predictions[i] - y_test[i]) for i in xrange(len(predictions))]
            accuracy = np.median(error)
        return accuracy
    
    def categorize(self, labels):
        mapping = {}
        labels = pd.Series(labels)
        for i,v in enumerate(list(set(labels))):
            labels = labels.replace(v, i)
            mapping[i] = v
        return np.array(labels), mapping  
    
    def stacker(self):
        self.stacking = True
        self.train_stacker()
        self.stacking = False
        return self.test_stacker()
    
    def time_elapsed(self):
        seconds = time.time()-self.time
        hours = int(seconds/3600)
        seconds = seconds-hours*3600
        minutes = int(seconds/60)
        seconds = seconds - minutes*60
        print "Time Elapsed = %dh %dm %ds" %(hours, minutes, seconds)
        
    def mean(self, l):
        means = []
        for i in xrange(len(l[0])):
            s = 0
            for j in l:
                s += j[i]
            means.append(float(s)/len(l))
        return means
    
    def preprocess(self, x):
        data = pd.DataFrame(x)
        for i in data:
            data[i] = data[i] - data[i].mean()
            data[i] = data[i]/float(data[i].std())
        return data

    
    def feature_transform(self, x):
        x = self.preprocess(x)
        scale = sklearn.preprocessing.MinMaxScaler()
        return np.column_stack((x, np.exp(x), np.sqrt(scale.fit_transform(x))))
        
    
    def train_stacker(self):
        for model in self.algorithms:
            generate_train, generate_test = model()
            if self.testing == False:
                self.stacking_train = np.column_stack((self.stacking_train, generate_train))
                self.stacking_test = np.column_stack((self.stacking_test, generate_test))
            else:
                self.test_stacking_train = np.column_stack((self.test_stacking_train, generate_train))
                self.test_stacking_test = np.column_stack((self.test_stacking_test, generate_test))
        return
        
    def test_stacker(self):
        if self.testing == False:
            return [model(self.stacking_train, self.y_train2, self.stacking_test, self.y_test) \
                for model in self.algorithms]
        else:
            return [model(self.test_stacking_train, self.y2, self.test_stacking_test, None)\
                for model in self.algorithms]
            
    def initialize_data(self, x_train=None, y_train=None, x_test=None, y_test=None):
        if self.testing == False:
            if self.stacking == False:
                return self.x_train, self.y_train, self.x_test, self.y_test
            else:
                if x_train == None:
                    x_train = self.x_train1
                if y_train == None:
                    y_train = self.y_train1
                if x_test == None:
                    x_test = self.x_train2
                if y_test == None:
                    y_test = self.y_train2
                return x_train, y_train, x_test, y_test
        else:
            if self.stacking == False:
                return self.x, self.y, self.test, None
            else:
                if x_train == None:
                    x_train = self.x1
                if y_train == None:
                    y_train = self.y1
                if x_test == None:
                    x_test = self.x2
                if y_test == None:
                    y_test = self.y2
                return x_train, y_train, x_test, y_test
    
    def stacking_test_data(self):
        if self.testing == False:
            return self.x_test
        else:
            return self.test
        
    def _linear_regression(self, x_train, y_train, x_test):
        lr = linear_model.LinearRegression()
        lr = lr.fit(x_train, y_train)
        return lr.predict(x_test)
    
    def linear_regression(self, x_train=None, y_train=None, x_test=None, y_test=None):
        x_train, y_train, x_test, y_test = self.initialize_data(x_train, y_train, x_test, y_test)
        linear_predictions = self._linear_regression(x_train, y_train, x_test)
        if self.stacking == True:
            return linear_predictions, self._linear_regression(x_train, y_train, self.stacking_test_data())
        return linear_predictions
    
    def _ridge_regression(self, x_train, y_train, x_test):
        lr = linear_model.Ridge()
        lr = lr.fit(x_train, y_train)
        return lr.predict(x_test)
    
    def ridge_regression(self, x_train=None, y_train=None, x_test=None, y_test=None):
        x_train, y_train, x_test, y_test = self.initialize_data(x_train, y_train, x_test, y_test)
        linear_predictions = self._ridge_regression(x_train, y_train, x_test)
        if self.stacking == True:
            return linear_predictions, self._ridge_regression(x_train, y_train, self.stacking_test_data())
        return linear_predictions
    
    def _lasso_regression(self, x_train, y_train, x_test):
        lr = linear_model.Lasso()
        lr = lr.fit(x_train, y_train)
        return lr.predict(x_test)
    
    def lasso_regression(self, x_train=None, y_train=None, x_test=None, y_test=None):
        x_train, y_train, x_test, y_test = self.initialize_data(x_train, y_train, x_test, y_test)
        linear_predictions = self._lasso_regression(x_train, y_train, x_test)
        self.linear_preds = linear_predictions
        if self.stacking == True:
            return linear_predictions, self._lasso_regression(x_train, y_train, self.stacking_test_data())
        return linear_predictions
        
    def xgboost(self, x_train=None, y_train=None, x_test=None, y_test=None):
        x_train, y_train, x_test, y_test = self.initialize_data(x_train, y_train, x_test, y_test)
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test)
        if self.task == 'classification':
            param = {'bst:max_depth':50, 'bst:eta':1, 'silent':1,'objective':'multi:softmax'}
#         elif self.task == 'regression':
#             param = {'bst:max_depth':50, 'bst:eta':1, 'silent':1,'objective':'reg:linear'}
        param['num_class'] = x_train.shape[1]
#         evallist = [(dtest,'eval'), (dtrain,'train')]
        num_round = 10
        bst = xgb.train(param, dtrain, num_round, verbose_eval=False)
        preds = bst.predict(dtest)
        self.xgb_predictions = preds
#         for i in xrange(len(labels)):
#             print labels[i], self.y_test[i]
        if self.stacking == True:
            return preds, bst.predict(xgb.DMatrix(self.stacking_test_data()))
        return preds
    
    def gb(self, x_train=None, y_train=None, x_test=None, n_estimators=400, max_depth=2):
        if self.task == 'classification':
            booster = ensemble.GradientBoostingClassifier(n_estimators = n_estimators, max_depth = max_depth)
        elif self.task == 'regression':
            booster = ensemble.GradientBoostingRegressor(n_estimators = n_estimators, max_depth = max_depth)  
        booster = booster.fit(x_train, y_train)
        return booster.predict(x_test)
    
    def gradient_booster(self, x_train=None, y_train=None, x_test=None, y_test=None):
        x_train, y_train, x_test, y_test = self.initialize_data(x_train, y_train, x_test, y_test)
        if self.hyperparameters == 'default':
            booster_predictions = self.gb(x_train, y_train, x_test)
            if self.stacking == True:
                return booster_predictions, self.gb(x_train, y_train, self.stacking_test_data())
            return booster_predictions
        elif self.hyperparameters == 'individual':
            if self.gb_n_estimators == None:
                max_accuracy = -1
                for n_estimators in xrange(100, 1300, 300):
                    booster_predictions = self.gb(x_train, y_train, x_test, n_estimators=n_estimators)
                    if self.accuracy(booster_predictions) > max_accuracy:
                        max_accuracy = self.accuracy(booster_predictions)
                        self.gb_n_estimators = n_estimators
            if self.gb_max_depth == None:
                max_accuracy = -1
                for max_depth in xrange(2,15,3):
                    booster_predictions = self.gb(x_train, y_train, x_test, max_depth = max_depth)
                    if self.accuracy(booster_predictions) > max_accuracy:
                        max_accuracy = self.accuracy(booster_predictions)
                        self.gb_max_depth = max_depth
                print "Optimal Booster Hyperparams: n_estimators = %d, max_depth = %d" \
                        %(self.gb_n_estimators, self.gb_max_depth)
            booster_predictions = self.gb(x_train, y_train, x_test, n_estimators=self.gb_n_estimators,\
                                          max_depth=self.gb_max_depth)
            if self.stacking == True:
                return booster_predictions, self.gb(x_train, y_train, self.stacking_test_data())
            return booster_predictions
        elif self.hyperparameters == 'grid':
            max_accuracy = -1
            for n_estimators in xrange(100, 1000, 300):
                for max_depth in xrange(2,15,3):
                    booster_predictions = self.gb(x_train, y_train, x_test, n_estimators=n_estimators, max_depth = max_depth)
                    print str(self.accuracy(booster_predictions)) + ' ' + str(n_estimators) + ' ' + str(max_depth)
                    if self.accuracy(booster_predictions) > max_accuracy:
                        max_accuracy = self.accuracy(booster_predictions)
                        max_predictions = booster_predictions
                        optimal_parameters = [n_estimators, max_depth]
            print("Gradient booster: ", optimal_parameters)
            return max_predictions
                
    def rf(self, x_train, y_train, x_test, n_estimators=400, max_features="auto"):
        if self.task == 'classification':
            forest = ensemble.RandomForestClassifier(n_estimators=n_estimators, max_features=max_features)
        elif self.task == 'regression':
            forest = ensemble.RandomForestRegressor(n_estimators=n_estimators, max_features=max_features)
        forest = forest.fit(x_train, y_train)
        return forest.predict(x_test)            
        
    
    def random_forest(self, x_train=None, y_train=None, x_test=None, y_test=None):
        x_train, y_train, x_test, y_test = self.initialize_data(x_train, y_train, x_test, y_test)
        if self.hyperparameters == 'default':
            if self.stacking == True:
                return self.rf(x_train, y_train, x_test), self.rf(x_train, y_train, self.stacking_test_data())
            return self.rf(x_train, y_train, x_test)
        elif self.hyperparameters == 'individual':
            if self.rf_n_estimators == None:
                max_accuracy = -1
                for n_estimators in xrange(100, 1400, 400):
                    forest_predictions = self.rf(x_train, y_train, x_test, n_estimators=n_estimators)
                    if self.accuracy(forest_predictions) > max_accuracy:
                        max_accuracy = self.accuracy(forest_predictions)
                        self.rf_n_estimators = n_estimators
            if self.rf_max_features == None:
                feat = x_train.shape[1]
                max_accuracy = -1
                for max_features in xrange(int(.3*feat), int(.6*feat), int(.05*feat)+1):
                    print max_features, feat
                    if max_features == 0:
                        max_features = 1
                    forest_predictions = self.rf(x_train, y_train, x_test, max_features=max_features)
                    if self.accuracy(forest_predictions) > max_accuracy:
                        max_accuracy = self.accuracy(forest_predictions)
                        self.rf_max_features = max_features
                print "Optimal Forest Hyperparams: n_estimators = %d, max_features = %d" \
                        %(self.rf_n_estimators, self.rf_max_features)
            forest_predictions = self.rf(x_train, y_train, x_test,\
                 n_estimators=self.rf_n_estimators, max_features = self.rf_max_features)
            if self.stacking == True:
                return forest_predictions, self.rf(x_train, y_train, self.stacking_test_data(),\
                                                   self.rf_n_estimators, self.rf_max_features)
            return forest_predictions 
        elif self.hyperparameters == 'grid':
            feat = x_train.shape[1]
            max_accuracy = -1
            for n_estimators in xrange(100, 1000, 200):
                for max_features in xrange(int(.3*feat), int(.6*feat), int(.03*feat)+1):
                    if max_features == 0:
                        max_features = 1
                    for criterion in ['gini', 'entropy']:
                        forest_predictions = self.rf(x_train, y_train, x_test, n_estimators = \
                                 n_estimators, criterion = criterion, max_features = max_features)
                        print self.accuracy(forest_predictions)
                        if self.accuracy(forest_predictions) > max_accuracy:
                            max_accuracy = self.accuracy(forest_predictions)
                            max_predictions = forest_predictions
                            optimal_parameters = [n_estimators, max_features, criterion]
            print["Random forest: ", optimal_parameters]
            return max_predictions

    def erf(self, x_train, y_train, x_test, n_estimators=400, max_features="auto"):
        if self.task == 'classification':
            forest = ensemble.ExtraTreesClassifier(n_estimators=n_estimators, max_features=max_features)
        elif self.task == 'regression':
            forest = ensemble.ExtraTreesRegressor(n_estimators=n_estimators, max_features=max_features)
        forest = forest.fit(x_train, y_train)
        return forest.predict(x_test)         
    
    def extremely_random_forest(self, x_train=None, y_train=None, x_test=None, y_test=None):
        x_train, y_train, x_test, y_test = self.initialize_data(x_train, y_train, x_test, y_test)
        if self.hyperparameters == 'default':
            if self.stacking == True:
                return self.erf(x_train, y_train, x_test), self.erf(x_train, y_train, self.stacking_test_data())
            return self.erf(x_train, y_train, x_test)
        elif self.hyperparameters == 'individual':
            if self.erf_n_estimators == None:
                max_accuracy = -1
                for n_estimators in xrange(100, 1300, 400):
                    erf_predictions = self.erf(x_train, y_train, x_test, n_estimators = n_estimators)
                    if self.accuracy(erf_predictions) > max_accuracy:
                        max_accuracy = self.accuracy(erf_predictions)
                        self.erf_n_estimators = n_estimators
            if self.erf_max_features == None:
                feat = x_train.shape[1]
                max_accuracy = -1
                for max_features in xrange(int(.3*feat), int(.6*feat), int(.05*feat)+1):
                    erf_predictions = self.erf(x_train, y_train, x_test, max_features=max_features)
                    if self.accuracy(erf_predictions) > max_accuracy:
                        max_accuracy = self.accuracy(erf_predictions)
                        self.erf_max_features = max_features
                print "Optimal erf Hyperparams: n_estimators = %d, max_features = %d" \
                        %(self.erf_n_estimators, self.erf_max_features)
            erf_predictions = self.erf(x_train, y_train, x_test,\
                 n_estimators=self.erf_n_estimators, max_features = self.erf_max_features)
            if self.stacking == True:
                return erf_predictions, self.erf(x_train, y_train, self.stacking_test_data(),\
                         n_estimators=self.erf_n_estimators, max_features = self.erf_max_features)
            return erf_predictions
    
    def logistic_regression(self, x_train=None, y_train=None, x_test=None, y_test=None):
        x_train, y_train, x_test, y_test = self.initialize_data(x_train, y_train, x_test, y_test)
        if self.hyperparameters == 'default':
            logistic = linear_model.LogisticRegression(penalty="l2", C=1)#class_weight='auto'
            logistic.fit(x_train, y_train)
            logistic_predictions = logistic.predict(x_test)
            if self.stacking == True:
                return logistic_predictions, logistic.predict(self.stacking_test_data())
            return logistic_predictions
        elif self.hyperparameters == 'individual':
            if self.log_reg_penalty == None:
                for penalty in ['l1', 'l2']:
                    if penalty == 'l1':
                        logistic = linear_model.LogisticRegression(penalty="l1")#class_weight='auto'
                        logistic.fit(x_train, y_train)
                        logistic_predictions = logistic.predict(x_test)
                        l1_accuracy = self.accuracy(logistic_predictions)
                        self.log_reg_penalty = 'l1'
                    else:
                        logistic = linear_model.LogisticRegression(penalty="l2")#class_weight='auto'
                        logistic.fit(x_train, y_train)
                        logistic_predictions = logistic.predict(x_test)
                        if self.accuracy(logistic_predictions) > l1_accuracy:
                            self.log_reg_penalty = 'l2'
            if self.log_reg_C == None:
                max_accuracy = -1
                for C in xrange(1,104,4):
                    logistic = logistic = linear_model.LogisticRegression(penalty=self.log_reg_penalty, C=C)
                    logistic.fit(x_train, y_train)
                    logistic_predictions = logistic.predict(x_test)
                    if self.accuracy(logistic_predictions) > max_accuracy:
                        max_accuracy = self.accuracy(logistic_predictions)
                        self.log_reg_C = C
                print "Optimal Logistic Regression Hyperparams: Penalty = %s, C = %d."\
                    %(self.log_reg_penalty, self.log_reg_C)
            logistic = linear_model.LogisticRegression(penalty=self.log_reg_penalty, C = self.log_reg_C)#class_weight='auto'
            logistic.fit(x_train, y_train)
            logistic_predictions = logistic.predict(x_test)
            if self.stacking == True:
                return logistic_predictions, logistic.predict(self.stacking_test_data())
            return logistic_predictions
    
    def lda(self, x_train=None, y_train=None, x_test=None, y_test=None):
        x_train, y_train, x_test, y_test = self.initialize_data(x_train, y_train, x_test, y_test)
        lda = discriminant_analysis.LinearDiscriminantAnalysis()
        lda.fit(x_train, y_train)
        lda_predictions = lda.predict(x_test)
        if self.stacking == True:
            return lda_predictions, lda.predict(self.stacking_test_data())
        return lda_predictions
    
    def qda(self, x_train=None, y_train=None, x_test=None, y_test=None):
        x_train, y_train, x_test, y_test = self.initialize_data(x_train, y_train, x_test, y_test)
        if self.hyperparameters == 'default':
            qda = discriminant_analysis.QuadraticDiscriminantAnalysis()
            qda.fit(x_train, y_train)
            qda_predictions = qda.predict(x_test)
            if self.stacking == True:
                return qda_predictions, qda.predict(self.stacking_test_data())            
            return qda_predictions
        elif self.hyperparameters == 'individual':
            if self.qda_reg_param == None:
                max_accuracy = -1
                for reg_param in [x*.1 for x in range(0,10,2)]:
                    qda = discriminant_analysis.QuadraticDiscriminantAnalysis(reg_param = reg_param)
                    qda.fit(x_train, y_train)
                    qda_predictions = qda.predict(x_test)
                    if self.accuracy(qda_predictions) > max_accuracy:
                        max_accuracy = self.accuracy(qda_predictions)
                        self.qda_reg_param = reg_param
                print "Optimal QDA Hyperparams: Reg_param = %d" %(self.qda_reg_param)
            qda = discriminant_analysis.QuadraticDiscriminantAnalysis(reg_param = self.qda_reg_param)
            qda.fit(x_train, y_train)
            qda_predictions = qda.predict(x_test)
            if self.stacking == True:
                return qda_predictions, qda.predict(self.stacking_test_data())
            return qda_predictions
    
    def svc(self, x_train=None, y_train=None, x_test=None, y_test=None):
        x_train, y_train, x_test, y_test = self.initialize_data(x_train, y_train, x_test, y_test)
        if self.hyperparameters == 'default':
            support_vm = svm.SVC()
            support_vm.fit(x_train, y_train)
            svm_predictions = support_vm.predict(x_test)
            if self.stacking == True:
                return svm_predictions, s.predict(self.stacking_test_data())
            return svm_predictions
        elif self.hyperparameters == 'individual':
            if self.svc_C == None:
                max_accuracy = -1
                for C in [10, 100]:
                    s = svm.SVC(C=C)
                    s.fit(x_train, y_train)
                    svm_predictions = s.predict(x_test)
                    if self.accuracy(svm_predictions) > max_accuracy:
                        max_accuracy = self.accuracy(svm_predictions)
                        self.svc_C = C
            if self.svc_kernel == None:
                max_accuracy = -1
                for kernel in ['rbf', 'linear', 'poly']:
                    s = svm.SVC(C = self.svc_C, kernel=kernel)
                    s.fit(x_train, y_train)
                    svm_predictions = s.predict(x_test)
                    if self.accuracy(svm_predictions) > max_accuracy:
                        max_accuracy = self.accuracy(svm_predictions)
                        self.svc_kernel = kernel
                print "Optimal SVM Hyperparams: Kernel = %s, C = %d" %(self.svc_kernel, self.svc_C)
            s = svm.SVC(C = self.svc_C, kernel=self.svc_kernel)
            s.fit(x_train, y_train)
            svm_predictions = s.predict(x_test)
            if self.stacking == True:
                return svm_predictions, s.predict(self.stacking_test_data())
            return svm_predictions
    
    def blend(self, args):
        blended_predictions = []
        if self.task == 'classification':
            n = len(args)
            for i in xrange(len(args[0])):
                counts = {}
                for j in xrange(n):
                    try:
                        counts[args[j][i]] += 1
                    except KeyError:
                        counts[args[j][i]] = 1
                maximus = -float('inf')
                for index, value in counts.iteritems():
                    if value > maximus:
                        max_value = index
                        maximus = value
                blended_predictions.append(max_value)
        elif self.task == 'regression':
            blended_predictions = self.mean(args)
        return blended_predictions
    
    def run_algorithms(self):
        if self.task == 'classification':
            return [i() for i in self.algorithms]
        elif self.task == 'regression':
            return [i() for i in self.algorithms]
    
    def create_predictions(self):
        self.testing = True
        predictions = (self.run_algorithms() + self.stacker())
        self.testing = False
        
        predictions_df = pd.DataFrame(predictions).transpose()
        self.testing_df = predictions_df
        predictions_df.columns = self.algorithm_names
        
        optimal_predictions = self.blend([predictions_df[j].values for j in self.optimal_iterator])
        self.time_elapsed()
        return optimal_predictions
    
    def predict(self):
        algorithms = self.run_algorithms()

        for i, v in enumerate(algorithms):
            print self.accuracy(v), self.algorithm_names[i]
        
        predictions = (algorithms + self.stacker())
        
        predictions_df = pd.DataFrame(predictions).transpose()
        self.training_df = predictions_df
        predictions_df.columns = self.algorithm_names

        combinations = []     
        if self.task == 'classification':
            col = predictions_df.columns
            for i in xrange(1, len(col), 2):
                for j in itertools.combinations(col, i):
                    combinations.append([k for k in j])            
            maximus = -1
            for i in combinations:
                blended = self.blend([predictions_df[j].values for j in i])
                accuracy = self.accuracy(blended)
                if accuracy > maximus:
                    maximus = accuracy
                    optimal_predictions = blended
                    self.optimal_iterator = i
                    print str(accuracy) + str(i)            
        elif self.task == 'regression':
            col = predictions_df.columns
            for i in xrange(1, len(col)):
                for j in itertools.combinations(col, i):
                    combinations.append([k for k in j])            
            maximus = float("inf")
            for i in combinations:
                blended = self.blend([predictions_df[j].values for j in i])
                accuracy = self.accuracy(blended)
                if accuracy < maximus:
                    maximus = accuracy
                    optimal_predictions = blended
                    self.optimal_iterator = i
                    print str(accuracy) + str(i)
        
        if self.test != None: 
            return self.create_predictions()
        else:
            self.time_elapsed()
            return optimal_predictions