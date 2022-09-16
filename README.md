# analytical-tools :eyeglasses:

Additional Information
---

Feature Importances 

*Random Forest*
There are indeed several ways to get feature "importances". As often, there is no strict consensus about what this word means. 

Scikit-Learn implements the importance as described in [1], called the "gini importance" or "mean decrease impurity". It is defined as the total decrease in node impurity (weighted by the probability of reaching that node (which is approximated by the proportion of samples reaching that node)) averaged over all trees of the ensemble.  

In the literature or in some other packages, you can also find feature importances implemented as the "mean decrease accuracy". Basically, the idea is to measure the decrease in accuracy on OOB data when you randomly permute the values for that feature. If the decrease is low, then the feature is not important, and vice-versa.

(Note that both algorithms are available in the randomForest R package.)

In RandomForestClassifier, the estimators_ attribute is a list of DecisionTreeClassifier (as mentioned in the documentation). In order to compute the feature_importances_ for the RandomForestClassifier, in scikit-learn's source code, it averages over all estimator's (all DecisionTreeClassifer's) feature_importances_ attributes in the ensemble.  

In DecisionTreeClassifer's documentation, it is mentioned that "The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance."[2]  

[1] L. Breiman, and A. Cutler, [“Random Forests”](http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm)  
[2]: Breiman, Friedman, "Classification and regression trees", 1984.
---

Statsmodels Results Summary  

*Log-Likelihood* 
- This is the maximized value of the log-likelihood function.  

*LL-Null*  
- This is the result of the maximized log-likelihood function when only an intercept is included.  
- It forms the basis for the pseudo-R^2 statistic and the Log-Likelihood Ratio (LRR) test.  

*Pseudo-R^2*
- This is a substitute of the familiar R^2 available under least squares.  
- It is computed based on the ratio of the maximized log-likelihood function for the null model m0 and the full model m1.


*Omnibus* 
  - A test of the skewness and kurtosis of the residual (characteristic #2). 
  - Goal: value close to zero which would indicate normalcy. 
  
*Prob(Omnibus)*  
  - Performs a statistical test indicating the probability that the residuals are normally distributed.  
  - Goal: something close to 1 here.  
  
*Skew*  
  - A measure of data symmetry.  
  - Goal: Something close to zero, indicating the residual distribution is normal.  
  - Note that this value also drives the Omnibus. 
  
*Kurtosis*  
  - A measure of "peakiness", or curvature of the data. 
  - Higher peaks lead to greater Kurtosis. 
  - Greater Kurtosis can be interpreted as a tighter clustering of residuals around zero, implying a better model with few outliers.  
  
*Durbin-Watson** 
  - Tests for homoscedasticity. 
  - Goal: value between 1 and 2. 
  
*Jarque-Bera (JB)/Prob(JB)* 
  - Similar to the Omnibus test in that it tests both skew and kurtosis. 
  - Goal: confirmation of the Omnibus test. 
  
*Condition Number*  
  - This test measures the sensitivity of a function's output as compared to its input.  
  - When we have multicollinearity, we can expect much higher fluctuations to small changes in the data.  
  - Goal: relatively small number, something below 30.   
  ---

References

B. Seligman, S. Tuljapurkar, D. Rehkopf  
  Machine learning approaches to the social determinants of health in the health and retirement study.
  SSM - Population Health, 4 (2018), pp. 95-99, [10.1016/j.ssmph.2017.11.008](https://www.sciencedirect.com/science/article/pii/S2352827317302331?via%3Dihub)



