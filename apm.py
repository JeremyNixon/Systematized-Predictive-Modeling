import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

class classifier():
    def __init__(self, x, y, test=None, hyperparameters='individual'):
        self.time = time.time()
        self.hyperparameters = hyperparameters
        if type(y[0]) != int:
            y, self.mapping = self.categorize(y)
        self.x = np.array(x)
        self.y = np.array(y, dtype = int)
        if test is None:
            self.test = None
        else:
            self.test = np.array(test)            
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
        
        self.stacking = False
        self.testing = False
        self.optimal_iterator = None
        
        self.algorithms = [self.xgboost, self.gradient_booster, self.random_forest,\
              self.extremely_random_forest, self.logistic_regression, self.lda, self.qda]
        
        self.xgb_max_depth = None
        self.gb_n_estimators = None
        self.gb_max_depth = None
        self.rf_n_estimators = None
        self.rf_max_features = None
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
        count = 0
        for i in xrange(len(predictions)):
            if predictions[i] == y_test[i]:
                count += 1
        return count/float(len(predictions))
    
    def categorize(self, labels):
        mapping = {}
        labels = pd.Series(labels)
        for i,v in enumerate(list(set(labels))):
            labels = labels.replace(v, i)
            mapping[i] = v
        return labels, mapping  
    
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
    
    def xgboost(self, x_train=None, y_train=None, x_test=None, y_test=None):
        x_train, y_train, x_test, y_test = self.initialize_data(x_train, y_train, x_test, y_test)
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test)
        param = {'bst:max_depth':50, 'bst:eta':1, 'silent':1,'objective':'multi:softmax'}
        param['num_class'] = x_train.shape[1]
#         evallist = [(dtest,'eval'), (dtrain,'train')]
        num_round = 10
        bst = xgb.train(param, dtrain, num_round, verbose_eval=False)
        preds = bst.predict(dtest)
#         for i in xrange(len(labels)):
#             print labels[i], self.y_test[i]
        if self.stacking == True:
            return preds, bst.predict(xgb.DMatrix(self.stacking_test_data()))
        return preds
    
    def gradient_booster(self, x_train=None, y_train=None, x_test=None, y_test=None):
        x_train, y_train, x_test, y_test = self.initialize_data(x_train, y_train, x_test, y_test)
        if self.hyperparameters == 'default':
            booster = ensemble.GradientBoostingClassifier(n_estimators = 400, max_depth = 2)
            booster = booster.fit(x_train, y_train)
            booster_predictions = booster.predict(x_test)
            if self.stacking == True:
                return booster_predictions, booster.predict(self.stacking_test_data())
            return booster_predictions
        elif self.hyperparameters == 'individual':
            if self.gb_n_estimators == None:
                max_accuracy = -1
                for n_estimators in xrange(100, 1300, 300):
                    booster = ensemble.GradientBoostingClassifier\
                    (n_estimators = n_estimators)
                    booster = booster.fit(x_train, y_train)
                    booster_predictions = booster.predict(x_test)
                    if self.accuracy(booster_predictions) > max_accuracy:
                        max_accuracy = self.accuracy(booster_predictions)
                        self.gb_n_estimators = n_estimators
            if self.gb_max_depth == None:
                max_accuracy = -1
                for max_depth in xrange(2,15,3):
                    booster = ensemble.GradientBoostingClassifier\
                    (max_depth = max_depth)
                    booster = booster.fit(x_train, y_train)
                    booster_predictions = booster.predict(x_test)
                    if self.accuracy(booster_predictions) > max_accuracy:
                        max_accuracy = self.accuracy(booster_predictions)
                        self.gb_max_depth = max_depth
                print "Optimal Booster Hyperparams: n_estimators = %d, max_depth = %d" \
                        %(self.gb_n_estimators, self.gb_max_depth)
            booster = ensemble.GradientBoostingClassifier(\
                          n_estimators = self.gb_n_estimators,
                          max_depth = self.gb_max_depth)
            booster = booster.fit(x_train, y_train)
            booster_predictions = booster.predict(x_test)
            if self.stacking == True:
                return booster_predictions, booster.predict(self.stacking_test_data())
            return booster_predictions
        elif self.hyperparameters == 'grid':
            max_accuracy = -1
            for n_estimators in xrange(100, 1000, 300):
                for max_depth in xrange(2,15,3):
                    booster = ensemble.GradientBoostingClassifier\
                    (n_estimators = n_estimators, max_depth = max_depth)
                    booster = booster.fit(x_train, y_train)
                    booster_predictions = booster.predict(x_test)
                    print str(self.accuracy(booster_predictions)) + ' ' + str(n_estimators) + ' ' + str(max_depth)
                    if self.accuracy(booster_predictions) > max_accuracy:
                        max_accuracy = self.accuracy(booster_predictions)
                        max_predictions = booster_predictions
                        optimal_parameters = [n_estimators, max_depth]
            print("Gradient booster: ", optimal_parameters)
            return max_predictions, 
                
    
    def random_forest(self, x_train=None, y_train=None, x_test=None, y_test=None):
        x_train, y_train, x_test, y_test = self.initialize_data(x_train, y_train, x_test, y_test)
        if self.hyperparameters == 'default':
            forest = ensemble.RandomForestClassifier(n_estimators=100)
            forest = forest.fit(x_train, y_train)
            forest_predictions = forest.predict(x_test)
            return forest_predictions
        elif self.hyperparameters == 'individual':
            if self.rf_n_estimators == None:
                max_accuracy = -1
                for n_estimators in xrange(100, 1400, 400):
                    forest = ensemble.RandomForestClassifier(n_estimators = n_estimators)
                    forest = forest.fit(x_train, y_train)
                    forest_predictions = forest.predict(x_test)
                    if self.accuracy(forest_predictions) > max_accuracy:
                        max_accuracy = self.accuracy(forest_predictions)
                        self.rf_n_estimators = n_estimators
            if self.rf_max_features == None:
                feat = x_train.shape[1]
                max_accuracy = -1
                for max_features in xrange(int(.3*feat), int(.6*feat), int(.05*feat)+1):
                    forest = ensemble.RandomForestClassifier(max_features = max_features)
                    forest = forest.fit(x_train, y_train)
                    forest_predictions = forest.predict(x_test)
                    if self.accuracy(forest_predictions) > max_accuracy:
                        max_accuracy = self.accuracy(forest_predictions)
                        self.rf_max_features = max_features
                print "Optimal Forest Hyperparams: n_estimators = %d, max_features = %d" \
                        %(self.rf_n_estimators, self.rf_max_features)
            forest = ensemble.RandomForestClassifier(\
                 n_estimators=self.rf_n_estimators, max_features = self.rf_max_features)
            forest = forest.fit(x_train, y_train)
            forest_predictions = forest.predict(x_test)
            if self.stacking == True:
                return forest_predictions, forest.predict(self.stacking_test_data())
            return forest_predictions 
        elif self.hyperparameters == 'grid':
            feat = x_train.shape[1]
            max_accuracy = -1
            for n_estimators in xrange(100, 1000, 200):
                for max_features in xrange(int(.3*feat), int(.6*feat), int(.03*feat)+1):
                    for criterion in ['gini', 'entropy']:
                        forest = ensemble.RandomForestClassifier(n_estimators = \
                                 n_estimators, criterion = criterion, max_features = max_features)
                        forest = forest.fit(x_train, y_train)
                        forest_predictions = forest.predict(x_test)
                        print self.accuracy(forest_predictions)
                        if self.accuracy(forest_predictions) > max_accuracy:
                            max_accuracy = self.accuracy(forest_predictions)
                            max_predictions = forest_predictions
                            optimal_parameters = [n_estimators, max_features, criterion]
            print["Random forest: ", optimal_parameters]
            return max_predictions
                    
    
    def extremely_random_forest(self, x_train=None, y_train=None, x_test=None, y_test=None):
        x_train, y_train, x_test, y_test = self.initialize_data(x_train, y_train, x_test, y_test)
        if self.hyperparameters == 'default':
            erf = ensemble.ExtraTreesClassifier(n_estimators=100)
            erf = erf.fit(x_train, y_train)
            erf_predictions = erf.predict(x_test)
            return erf_predictions
        elif self.hyperparameters == 'individual':
            if self.erf_n_estimators == None:
                max_accuracy = -1
                for n_estimators in xrange(100, 1300, 400):
                    erf = ensemble.ExtraTreesClassifier(n_estimators = n_estimators)
                    erf = erf.fit(x_train, y_train)
                    erf_predictions = erf.predict(x_test)
                    if self.accuracy(erf_predictions) > max_accuracy:
                        max_accuracy = self.accuracy(erf_predictions)
                        self.erf_n_estimators = n_estimators
            if self.erf_max_features == None:
                feat = x_train.shape[1]
                max_accuracy = -1
                for max_features in xrange(int(.3*feat), int(.6*feat), int(.05*feat)+1):
                    erf = ensemble.ExtraTreesClassifier(max_features = max_features)
                    erf = erf.fit(x_train, y_train)
                    erf_predictions = erf.predict(x_test)
                    if self.accuracy(erf_predictions) > max_accuracy:
                        max_accuracy = self.accuracy(erf_predictions)
                        self.erf_max_features = max_features
                print "Optimal erf Hyperparams: n_estimators = %d, max_features = %d" \
                        %(self.erf_n_estimators, self.erf_max_features)
            erf = ensemble.ExtraTreesClassifier(\
                 n_estimators=self.erf_n_estimators, max_features = self.erf_max_features)
            erf = erf.fit(x_train, y_train)
            erf_predictions = erf.predict(x_test)
            if self.stacking == True:
                return erf_predictions, erf.predict(self.stacking_test_data())
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
        n = len(args)
        blended_predictions = []
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
        return blended_predictions
    
    def run_algorithms(self):
        xgboost_predictions = self.xgboost()
        booster_predictions = self.gradient_booster()
        forest_predictions = self.random_forest()
        erf_predictions = self.extremely_random_forest()
        logistic_predictions = self.logistic_regression()
        lda_predictions = self.lda()
        qda_predictions = self.qda()
        return [xgboost_predictions, booster_predictions, forest_predictions, erf_predictions,\
                  logistic_predictions, lda_predictions, qda_predictions]
    
    def create_predictions(self):
        self.testing = True
        predictions = (self.run_algorithms() + self.stacker())
        self.testing = False
        
        predictions_df = pd.DataFrame(predictions).transpose()
        self.testing_df = predictions_df
        predictions_df.columns = ['xgboost_predictions','booster_predictions', 'forest_predictions', 'erf_predictions',\
                  'logistic_predictions', 'lda_predictions', 'qda_predictions', \
                   'xgboost_on_stack', 'booster_on_stack','forest_on_stack', \
                    'erf_on_stack', 'logistic_on_stack', 'lda_on_stack', 'qda_on_stack']
        
        optimal_predictions = self.blend([predictions_df[j].values for j in self.optimal_iterator])
        self.time_elapsed()
        return optimal_predictions
    
    def predict(self):
        algorithms = self.run_algorithms()

        models = ['xgboost_predictions','booster_predictions', 'forest_predictions', 'erf_predictions',\
                  'logistic_predictions', 'lda_predictions', 'qda_predictions']
        for i, v in enumerate(algorithms):
        
            print self.accuracy(v), models[i]
        
        predictions = (algorithms + self.stacker())
        
        predictions_df = pd.DataFrame(predictions).transpose()
        self.training_df = predictions_df
        predictions_df.columns = ['xgboost_predictions','booster_predictions', 'forest_predictions', 'erf_predictions',\
                  'logistic_predictions', 'lda_predictions', 'qda_predictions', \
                   'xgboost_on_stack', 'booster_on_stack', \
                    'forest_on_stack', 'erf_on_stack', 'logistic_on_stack', 'lda_on_stack', \
                    'qda_on_stack']

        combinations = []
        col = predictions_df.columns[:-1]
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
        
        if self.test != None: 
            return self.create_predictions()
        else:
            self.time_elapsed()
            return optimal_predictions