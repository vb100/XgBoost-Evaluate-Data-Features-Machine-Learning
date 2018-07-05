# -- coding utf-8 --
# XGBoost in simple way. Can be improved by Gradient Descent, setuping learning rate, etc.
# Date 20180620

#------------------------------------------------------------------------------
# Use Pandas to load the data in a dataframe
import pandas as pd
df = pd.read_excel('default of credit card clients.xls', header = 1, index_col = 0)

print('The shape of dataframe is {}.'.format(df.shape))

#  Clean dataset 
def clean_dataset(df)
    df.loc[~df.EDUCATION.isin([1, 2, 3, 4]), 'EDUCATION'] = pd.np.nan
    for i in [0, 2, 3, 4, 5, 6]
        df.loc[df['PAY_{}'.format(i)]  -1] = pd.np.nan
    df.dropna(inplace = True)
    
clean_dataset(df)
print('The shape of dataframe after data cleaning is {}.'.format(df.shape))

#  Process Categorical Features 
def process_categorical_features(df)
    dummies_education = pd.get_dummies(df.EDUCATION, prefix = 'EDUCATION', drop_first = True)
    dummies_marriage = pd.get_dummies(df.MARRIAGE, prefix = 'MARRIAGE', drop_first = True)
    df.drop(['EDUCATION', 'MARRIAGE'], axis = 1, inplace = True)
    return pd.concat([df, dummies_education, dummies_marriage], axis = 1)

df = process_categorical_features(df)
print('The shape of dataframe after data cleaning and generating Dummies variables is {}.'.format(df.shape))

#  Extract target from the features 
y = df['default payment next month']
X = df[[col for col in df.columns if col != 'default payment next month']]

#  Split dataset into train and test sets 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

print('Size of train dataset {} rows.'.format(X_train.shape[0]))
print('Size of test dataset {} rows.'.format(X_test.shape[0]))

#------------------------------------------------------------------------------
#  Ready to train!
import xgboost as xgb

#  Set-up XGBoost classifier 
classifier = xgb.sklearn.XGBClassifier(nthread = -1, seed = 42)

#  Train a model with default parameters 
classifier.fit(X_train, y_train)

#  Evaluation 
# Let's use our trained model to predict whether the customer from the test set will
# default or not and evaluate the accuracy of our model

predictions = classifier.predict(X_test)

print(pd.DataFrame(predictions, index = X_test.index, columns = ['Predicted default']).head())
print(pd.DataFrame(y_test).head())

#  Use Score XgBoost method to see how accurate our model is for all test customers 
print('Model Accuracy {.2f}%'.format(100  classifier.score(X_test, y_test)))
'''The problem we are working on here is unbalanced most of the people do not 
default payments and the customers who default are rare in our dataset. This 
means that by predicting that none of the customers will default, we could get 
a good accuracy too, even though the model would be useless. There are ways to 
prevent our model to make such mistakes.'''

#------------------------------------------------------------------------------
#  Handy XGBoost methods

%matplotlib inline
import matplotlib.pyplot as plt

# Plot Feature Importances 
plt.figure(figsize=(10,7))
xgb.plot_importance(classifier, ax = plt.gca())

# Plot Tress that were built by XGBoost
plt.figure(figsize=(18,15))
xgb.plot_tree(classifier, ax = plt.gca())

# Access the characteristics of the model
print('Number of boosting trees {}.'.format(classifier.n_estimators))
print('Max depth of trees {}.'.format(classifier.max_depth))
print('Objective function {}.'.format(classifier.objective))
