# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 09:09:41 2019

@author: d
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

class Node:
    def __init__(self,parent, total_error, fitted_error, intercept, slope=0, feature=None, split_value=None):
        self.parent=parent
        self.total_error=total_error
        self.fitted_error=fitted_error
        self.left=None
        self.right=None
        self.only_child=None
        self.slope=slope
        self.intercept=intercept
        self.feature=feature
        self.split_value=split_value
    def set_right(self,node, feature, split_value):
        self.right=node
        self.split_value=split_value
        self.feature=feature
    def set_left(self,node, feature, split_value):
        self.left=node
        self.split_value=split_value
        self.feature=feature
    def set_only_child(self,node):
        self.only_child=node

        
    def predict(self,x):
        #check for children, otherwise return prediction
#        print("PREDICTING", self.__dict__)
        if not self.only_child is None:
            return self.only_child.predict(x)
        if not (self.left is None and self.right is None):
            if x[self.feature]>self.split_value:
                return self.right.predict(x)
            return self.left.predict(x)            
        if self.feature==None:
            return self.intercept
        return self.slope*x[self.feature] + self.intercept

def SE(actual, pred):
    return ((actual-pred)**2)

class slopeTree:
    def __init__(self, loss_function=SE, min_samples_per_node = 20):
        self.loss=loss_function
        self.min_samples_per_node = min_samples_per_node

    def _search(self,X_sorted,y,parent, step=10):
        scores=[] # delta error, error, intercept, slope, feature, split_value, 
        # use parent error
#        print X_sorted
        for feature in X_sorted.keys():
            x=X_sorted[feature]
            if feature!=parent.feature or parent.slope==0:
                # fit regression
                result=sm.OLS(y,sm.add_constant(x)).fit()
                
                print(result.summary())
                regression_error = self.loss(y, result.fittedvalues)
                total_error=sum(regression_error)
                slope=result.params[0]
                intercept=result.params[1]
                scores.append((parent.total_error-total_error, regression_error, total_error, intercept, None, None, None, slope, feature, None, None, None))
                # split on whatever provides most error reduction in given leaf
#                print 'FIT REGRESSION SUCCESSFULLY'
#                print parent.error-regression_error
            
            for i in range(step,x.shape[0]-step, step):
                lower = x.iloc[:i]
                upper = x.iloc[i:]
                lower_data = y.loc[lower.index]
                lower_mean = lower_data.mean()
                lower_error = self.loss(lower_data, lower_mean)
                lower_total_error = sum(lower_error)
                lower_parent_error = sum(parent.fitted_error.loc[lower.index])
                upper_data = y.loc[upper.index]
                upper_mean = upper_data.mean()
                upper_error = self.loss(upper_data, upper_mean)        
                upper_total_error = sum(upper_error)
                upper_parent_error = sum(parent.fitted_error.loc[upper.index])
                print(feature, i, upper_error, lower_error)
            
                scores.append((lower_parent_error-lower_total_error, lower_error, lower_total_error, lower_mean, upper_error, upper_total_error, upper_mean, 0, feature, x.iloc[i], lower.index, upper.index))
                scores.append((upper_parent_error-upper_total_error, lower_error, lower_total_error, lower_mean, upper_error, upper_total_error, upper_mean, 0, feature, x.iloc[i], lower.index, upper.index))
        print scores
        print pd.DataFrame(scores).sort_values(0)
#        raw_input()
        delta_error, lower_error, lower_total_error, lower_mean, upper_error, upper_total_error, upper_mean, slope, feature, split_value, lower_indices, upper_indices = min(scores)
        
        if split_value!=None:
            right=Node(parent, upper_total_error, upper_error, upper_mean, slope, None, None) # should pick correct split
            left =Node(parent, lower_total_error, lower_error, lower_mean, slope, None, None)
            parent.set_right(right, feature, split_value)
            parent.set_left(left, feature, split_value)
            
            #subset data
            sort_upper = {}
            sort_lower = {}
            for feature, x in X_sorted.iteritems():
                sort_upper[feature] = x[upper_indices]
                sort_lower[feature] = x[lower_indices]
            if len(upper_indices)>=self.min_samples_per_node:
                self._search(sort_upper,y[upper_indices],right) # pass in sort 
            if len(lower_indices)>=self.min_samples_per_node:
                self._search(sort_lower,y[lower_indices],left)
        else:
            child = Node(parent, lower_error, lower_mean, slope, feature, split_value)
            parent.set_only_child(child)
            self._search(X_sorted, y, child)
    
    def fit(self,X,y):
        intercept = y.mean()
        fitted_error = self.loss(y,intercept)
        total_error = sum(fitted_error)
        self.root=Node(None,total_error,fitted_error,intercept)
        
        # make dict of sorted features?
        X_sorted={}
        for feat in X.columns:
            X_sorted[feat] = X[feat].sort_values()
        self._search(X_sorted,y,self.root)
        
        
        return self


#    def _predict_row(self,x,node):
#        if node.right==None and node.left==None:
#            return node.predict(x)
#        if x[node.feature]>node.split:
#            return self._predict_row(x,node.right)
#        return self._predict_row(node.left)
            
    def predict(self,X):
        pred=pd.Series(index=X.index)
        for i, x in X.iterrows():
#            print('PREDICTING',i,x)
            pred.loc[i]=self.root.predict(x)    #self._predict_row(x, self.root)
        return pred
        

def print_tree(root):
    if isinstance(root, Node):
        print(root.__dict__)
        if isinstance(root.right, Node):
            print("RIGHT")
            print_tree(root.right)
            print("LEFT")
            print_tree(root.left)
        else:
            print("ONLY CHILD")
            print_tree(root.only_child)
    else:
        print(root)
        
        
if __name__ == "__main__":
    import sys
    sys.setrecursionlimit(1500)


    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    import matplotlib.pyplot as plt
    
    x1=pd.Series(np.arange(0,np.pi*2, np.pi*2/360))
    y1=x1.apply(np.sin)+ np.random.normal(0,.2, len(x1))
    X1=x1.to_frame()
    
    x11=pd.Series(np.arange(0,np.pi*2, np.pi*2/360))
    x21=pd.Series(np.arange(-np.pi,np.pi, np.pi*2/360))
    X2=pd.concat([x11,x21], axis=1)
    y2=np.sin(x1*x2) + np.random.normal(0,.2, len(x1))

    x3 = pd.Series(range(360))
    y3 = pd.Series(index=x3.index)
    y3[:300] = x3[:300]
    y3[300:]=300
    y3+=np.random.normal(0,10, len(x1))
    X3 = x3.to_frame()
    
    
    
    data=[(X1,y1), (X2, y2), (X3,y3)]
    models = [DecisionTreeRegressor(min_samples_leaf=20), RandomForestRegressor(min_samples_leaf=20),slopeTree()]
    
    for X,y in data:
        plt.figure()
        y.plot()
        for m in models:
            try:
                reg = m.fit(X,y)
            except:
                continue
            pred= reg.predict(X)
            pd.Series(pred, index=y.index).plot()
        plt.show()
    # fit a sine wave
    st = slopeTree()
    
    st.fit(x.to_frame(),y) 
    
    print_tree(st.root)
    
    pred=st.predict(x.to_frame())    

    dtr = DecisionTreeRegressor(min_samples_leaf=20).fit(x.to_frame(), y)
    dp = dtr.predict(x.to_frame())
    
    y.plot()
    pred.plot()
    pd.Series(dp, index=y.index).plot()
    
#    # fit fake data
#
    st1 = slopeTree()
    
    st1.fit(X,y1) 
    
    print_tree(st1.root)
    
    pred=st1.predict(x.to_frame())    
    pred.plot()
    y1.plot()
    dtr = DecisionTreeRegressor(min_samples_leaf=10).fit(X, y1)
    dp = dtr.predict(X)
    rfr = RandomForestRegressor(min_samples_leaf=10).fit(X,y1)
    rp = rfr.predict(X)
    pd.Series(dp, index=y1.index).plot()
    pd.Series(rp, index=y1.index).plot()

    
    #hinge
    
    