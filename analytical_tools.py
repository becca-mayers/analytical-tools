#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rebecca
"""

#Analytical Tools
from sklearn.ensemble import RandomForestClassifier
from statsmodels.formula.api import logit
from IPython.display import HTML, display
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn import metrics
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import pandas as pd
import numpy as np

#%% What's Missing  

def whats_missing(df):
    ''' Generate a table of percentage of missing values per column in descending order.
    params: 
        df: pandas dataframe
    returns:
        tabulated HTML table
    '''
    
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    #sort missing values from most to least
    mis_val_percent = mis_val_percent.sort_values(ascending=False)
    
    #make printable dropped columns df
    mis_val_percent = mis_val_percent.reset_index()
    mvpcols = ['variable','percentage']
    mis_val_percent.columns = mvpcols
    mis_val_percent = mis_val_percent.sort_values(by='percentage',ascending=False)
    mis_val_percent = mis_val_percent[mis_val_percent.percentage>=0]

    output = display(HTML(tabulate(mis_val_percent, 
                          headers = mvpcols, 
                          tablefmt='html', 
                          showindex=False)))
    return output



#%% Drop Missing  

def drop_missing(df, percentage):
    ''' Drop columns missing more than x% of their values.

    params:
        df: pandas dataframe 
        percentage: percentage threshold for removal
    
    returns: 
        transformed dataframe
    '''
    
    #get original column naming conventions
    columns = df.columns
    
    #determine percentages missing per column
    percent_missing = df.isnull().sum() * 100 / len(df)

    #create a dataframe from the percent missing variable
    missing_value_df = pd.DataFrame({'column_name': columns,
                                 'percent_missing': percent_missing})

    #generate a list of the columns that will be dropped based on the percentage argument
    missing_drop = list(missing_value_df[missing_value_df.percent_missing>percentage].column_name)
    
    #make printable dropped columns df
    dropped_table = missing_value_df[missing_value_df.percent_missing>percentage]
    dropped_table = dropped_table.reset_index()
    dropped_table.columns = ['col1','variable','percentage']
    dropped_table = dropped_table.drop('col1',axis=1)
    dropped_table = dropped_table.sort_values(by='percentage',ascending=False)
    
    #print out a list of the columns that were dropped for the user
    print('Dropping columns missing greater than ' + str(percentage) + '% of their values.')
    print('Old shape: ' + str(df.shape[0]) + ' rows, ' + str(df.shape[1]) + ' columns')
      
    #drop the columns from the generated list/print out
    df = df.drop(missing_drop, axis=1)
    #provide transformation information to user
    print('New shape: ' + str(df.shape[0]) + ' rows, ' + str(df.shape[1]) + ' columns')
    print(str(len(dropped_table)) + ' variables were dropped.')
    print('------')
    print('\n Dropped Variables Table')
    display(HTML(tabulate(dropped_table, 
                          headers = ['variable','percentage'], 
                          tablefmt='html', 
                          showindex=False)))

    #return the newly reduced dataframe
    return df

def remaining_variables(df):
    '''Create a nicely formatted table of the remaining columns.
    
    params:
        df: pandas dataframe
    '''
    print('Current shape: ' + str(df.shape[0]) + ' rows, ' + str(df.shape[1]) + ' columns')
    print('------')
    print('\n Current Variables Table')
    columns_df = pd.DataFrame(df.columns.values.tolist(), columns = ['remaining_variables'])
    display(HTML(tabulate(columns_df, 
                          headers = ['remaining_variables'], 
                          tablefmt='html', 
                          showindex=False)))

#%% Reduce low numerical variance  

def reduce_low_numerical_variance(df):
    '''Drop variables with a variance less than 0.0475,equivalent to a binary variable 
    with at least 95% of values in one category.
    Note: the 0.0475 threshold is based on the methodology of Seligman et al.
    
    B. Seligman, S. Tuljapurkar, D. Rehkopf
    Machine learning approaches to the social determinants of health in the health and retirement study.
    SSM - Population Health, 4 (2018), pp. 95-99, 10.1016/j.ssmph.2017.11.008
    
    params: 
        df: pandas dataframe 
    
    returns:
        transformed dataframe
    '''
    
    print('Old shape: ' + str(df.shape[0]) + ' rows, ' + str(df.shape[1]) + ' columns')
   
    #get variance
    var_check = df.var()

    #get list of features with variance less than .0475
    var_df = var_check[var_check<.0475]
    var_list = var_check.index[var_check<.0475].tolist()
    
    #drop those features from the dataset
    df = df.drop(var_list, axis = 1)
    var_df = var_df.reset_index()
    cols = ['variable','percentage']
    var_df.columns = cols
    
    #make printable dropped columns df
    var_df = var_df.sort_values(by='percentage',ascending=False)
    
    #provide transformation information to user
    print('New shape: ' + str(df.shape[0]) + ' rows, ' + str(df.shape[1]) + ' columns')
    print(str(len(var_list)) + ' variables dropped.')
    print('------')
    print('\n Dropped Variables Table')
    display(HTML(tabulate(var_df, 
                          headers = cols, 
                          tablefmt='html', 
                          showindex=False)))

    return df

#%% Reduce low categorical variance

def reduce_low_categorical_variance(df):
    '''Reduces the dimensionality of the dataset by removing non-numerical 
    columns with less than 5% variance.    
    Note: The 5% threshold is based on the methodology of Seligman et al.
    
    B. Seligman, S. Tuljapurkar, D. Rehkopf
    Machine learning approaches to the social determinants of health in the health and retirement study.
    SSM - Population Health, 4 (2018), pp. 95-99, 10.1016/j.ssmph.2017.11.008
    
    params: 
        df: pandas dataframe 
    
    returns:
        transformed dataframe
    '''

    #get object dtypes
    odf = df.select_dtypes('object')

    #determine counts per column
    cat_counts = odf.apply(lambda x: x.value_counts()).T.stack() / len(odf)
    
    #make dataframe
    cat_counts_df = pd.DataFrame(cat_counts)
    
    #reset multi-level index
    cat_counts_df = cat_counts_df.reset_index()
    
    #column names for new dataframe
    cols = ['variable', 'category', 'percentage']
    cat_counts_df.columns = cols
    cat_counts_df = cat_counts_df.sort_values(by = 'percentage', ascending = True)
                     
    #generate a list of the columns that will be dropped based on the percentage argument
    low_variance_drop = list(cat_counts_df[cat_counts_df.percentage>.95].variable)
    
    #make printable dropped columns df
    dropped_cats = cat_counts_df[cat_counts_df.percentage>.95]
    dropped_cats = dropped_cats.reset_index()
    dropped_cats = dropped_cats.sort_values(by='percentage',ascending=False)
    dropped_cats['percentage'] = dropped_cats['percentage']*100
    
    print('Dataset shape was ' + str(df.shape[0]) + ' rows and ' + str(df.shape[1]) + ' columns.')
    #drop the columns from the generated list/print out
    df = df.drop(low_variance_drop, axis = 1)
    
    #provide transformation information to user
    print('New shape: ' + str(df.shape[0]) + ' rows, ' + str(df.shape[1]) + ' columns')
    print(str(len(dropped_cats)) + ' variables dropped.')
    print('------')
    print('\n Dropped Variables Table')
    
    display(HTML(tabulate(dropped_cats, 
                          headers = cols, 
                          tablefmt = 'html', 
                          showindex = False)))

    #return the newly reduced dataframe to the user
    return df

#%% Classification model with accuracy

#Just the model
def classification_model(df, outcome, n_estimators = 100):
    '''Generates a classification model 
    
    params:
        df: pandas dataframe
        outcome: str; outcome variable
    
    returns:
        model: unfitted classification model
    '''
    
    #Create predictor list
    predictors = df.columns.drop(outcome).tolist()
    
    #instantiate Random Forest Classifier
    model = RandomForestClassifier(n_estimators = n_estimators)
    
    
    #Fit the model:
    model.fit(df[predictors], df[outcome])
    
    #Make predictions on training set:
    predictions = model.predict(df[predictors])
    
    #Print accuracy
    accuracy = metrics.accuracy_score(predictions, df[outcome])
    print ("Model Accuracy : %s" % "{0:.3%}".format(accuracy))

    return model

#%% Feature importances
#numerical df x feature importances

def get_feature_importances(df, outcome, model):
    '''Generate the imporance of each predictor feature and show the top 50%
    
    params:
        df: pandas dataframe
        outcome: str; outcome variable
    '''
    
    #Create predictor list
    predictors = df.columns.drop(outcome).tolist()

    #Generate feature importances array
    importances = model.feature_importances_
    
    #Make feature importances dataframe
    feat_imp = pd.DataFrame(importances, index = predictors).reset_index()
    feat_imp.columns = ['feature','importance']
    
    #round importance values
    feat_imp['importance'] = feat_imp['importance'].round(3)*100
    
    #sort and set descending
    feat_imp = feat_imp.sort_values(by = 'importance', ascending = False)
    
    feat_imp = feat_imp.reset_index().drop('index', axis = 1)
    
    return feat_imp

#%%
def make_feature_importances_graph(feat_imp, outcome):
    '''Create a horizontal bar graph of the top 50% feature importances.
    
    params:
        feat_imp: pandas dataframe 
            Found using the get_feature_importances() function.
         outcome: str; outcome variable
    
    returns:
        fig: bar graph figure
    '''
    
    #sort and set descending
    feat_imp = feat_imp.sort_values(by = 'importance', ascending = False)
    
    #find the 50% integer
    #top_half_int = int(len(feat_imp.index)/2)
    #slice the top half of features accordingly
    #top_half_df = feat_imp[:top_half_int]

    #sort and set ascending
    #top_half_df = top_half_df.sort_values(by = 'importance', ascending = True)
    
    #greater than .2
    feat_over_point2 = feat_imp.loc[feat_imp['importance'] >= .2]
    feat_over_point2 = feat_over_point2.sort_values(by = 'importance', ascending = True)
    
    #generate horizontal bar graph
    fig = px.bar(feat_over_point2,
                 x = 'importance', 
                 y = 'feature',  
                 orientation = 'h',
                 hover_data = ['importance','feature'],
                 #height = 400,
                 title = 'Feature Importances >= .2: {}'.format(outcome))
    
    return fig.show()

#%%
def make_feature_importances_heatmap(feat_imp, outcome):
    '''Create heatmap of correlated numeric values.
    
    params:
        feat_imp: pandas dataframe
            Found using the get_feature_importances() function.
        outcome: str; outcome variable
    
    returns:
        fig: heatmap figure
    '''
    
    #Generate correlations
    corr_df = feat_imp.corr()
    
    #Generate figure
    fig = go.Figure(go.Heatmap(z = corr_df.values.tolist(),
                               x = corr_df.columns.tolist(),
                               y = corr_df.index.tolist(),
                               hoverongaps = False))
    
    #Generate layout
    layout_heatmap = go.Layout(
        title=('Correlating Feature Importances for {} Outcome'.format(outcome)))    
    fig.layout = layout_heatmap
    
    #Set colorbar
    fig.data[0].colorbar = dict(title='Correlation', titleside = 'top')
    #Set figure width & height
    fig.update_layout(width = 800, height = 700)
    
    return fig.show()

#%%
def logistic_regression_model(formula, df):
    '''Fit a logistic regression model (statsmodels) and return the model's summary.
    
    params:
        formula: patsy formula
            ex. "Q('Cesarean W Complications') ~ Q('African American/Black') + White";
            See https://patsy.readthedocs.io/en/latest/formulas.html
        
        df: pandas dataframe
    
    returns:
        model summary
    '''
    
    #fit model
    model = logit(formula, df).fit()
    
    return model.summary()

#%%
def logistic_regression_plot(x, y, df):
    '''Create a seaborn logistic regression plot. Confidence interval resampled
    500 times. Jitter (noise) added after fitting regression inly to influence the look of the
    scatter plot.
    
    params:
        x: str; x-axis
        y: str; y-axis. 
            Underlying variable must be binary.
        df: pandas dataframe
    
    returns:
        plt: plot figure
    '''
    
    #Generate regression plot
    ax = sns.regplot(x = x,
                     y = y,
                     data = df, 
                     logistic = True,
                     n_boot = 500, 
                     y_jitter = .03)
    
    #Set axis labels
    ax.set_ylabel(y)
    ax.set_xlabel(x)
    
    #Set title
    ax.set_title(x + ' & ' + y + ": Logistic Regression")
    
    return plt.show()

