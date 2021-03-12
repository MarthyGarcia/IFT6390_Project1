# Importing dependencies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# get_ipython().run_line_magic('matplotlib', 'inline')
from pandas.plotting import scatter_matrix
import warnings

warnings.filterwarnings('ignore')

# Model utils
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE

# Model selection
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate

# Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


# ------------ Step 0. Loading and visualizing the data ------------
url = "https://raw.githubusercontent.com/MarthyGarcia/IFT6390_Project1/main/Data/cancer_reg.csv"
df = pd.read_csv(url)

# Placing our label column at the end of the dataframe
df['label_deathrate'] = df['target_deathrate']
df.drop(['target_deathrate'], axis=1, inplace=True)
df.head()


# ------------ Step 1. Splitting the data into our train and test sets ------------
# Set seed
seed = 1000

# Separating our attributes from our labels
def split_feat_label(data):
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return x, y

x, y = split_feat_label(df)

# Splitting the data: test, train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=seed)

train = pd.concat([x_train, y_train], axis=1)
test = pd.concat([x_test, y_test], axis=1)


# ------------ Step 2. Exploring and analysing our data ------------
train.shape
train.info()

# The dataset contains 33 columns and the variable we look to predict is named: label_deathrate.
# Most variables are numerical with the exception of two categorial features named geography and binnedinc.

# Do we have any missing values?
train.isna().sum() / len(train)

# We can see that pctsomecole18_24 is missing almost 75.0% of its values, therefore it would be
# preferable to remove it. There is also pctprivatecoveragealone that is missing 20.0 % of its
# values and pctemployed16_over with 5.0% of NaN.

# We can set the missing values in "pctemployed16_over" and in "pctprivatecoveragealone" to some
# value (zero, the mean, the median, etc.). First let's look at their respective distributions.
plt.figure(figsize=(16, 6))
train["pctemployed16_over"].hist();

plt.figure(figsize=(16, 6))
train["pctprivatecoveragealone"].hist();

# Both variables seem to follow a normal distribution and are symetrical around the mean.
# Therefore we will look to replace those missing values with the mean.

# Deeper analysis.
train.describe()

# Looking at how the data is distributed.
train.hist(bins=50, figsize=(20, 15))
plt.show();

# From the plot above and the statistics report, it can be seen the some of the data is skewed and
# needs to be transformed.

# How correlated are our variables?
plt.figure(figsize=(24, 20))
sns.heatmap(train.corr(), vmax=0.6, square=True, annot=True);

# Let's pick the most relevant ones and generate a scatter matrix.
attributes = ["label_deathrate", "incidencerate", "pctpubliccoveragealone", "povertypercent", "pcths25_over",
              "pctpubliccoverage", "pctunemployed16_over"]
scatter_matrix(train[attributes], figsize=(12, 8));

# There seems to be a linear relationship between the target variable and most of the features.

# Let's analyse our first categorical variable: geography. We shall divide it by state as opposed to by counties.
train['geography'] = train['geography'].str.rsplit(', ').str[-1]
train['geography'].value_counts()

# Let's visualize those occurences per state.
plt.figure(figsize=(16, 6))
train['geography'].value_counts().plot(kind='bar');

# In a future version of this model, we could look to fetch more data from the states that don't contain
# as many occurrences. However, for the time being, we will remove the variable geography as the data is
# far too skeweed on observations per state. For instance, the state of Texas could have much higher cancer
# incidents and could make our model biased.


# ------------ Step 3. Feature engineering ------------
# Here we look to finalize cleaning the data and transforming the variables.
def feature_eng(data):
    # features to drop
    drop_features = ['pctsomecol18_24', 'geography']

    # remaining categorical variable
    categorical_features = ['binnedinc']

    # filter columns to drop
    every_feature = [col for col in data.columns if col not in drop_features]
    data = data[every_feature]

    # get numerical features
    every_column_non_categorical = [col for col in data.columns if col not in categorical_features]
    numeric_feats = data[every_column_non_categorical].dtypes[data.dtypes != "object"].index

    # apply log transformation to numerical variables
    data[numeric_feats] = np.log1p(data[numeric_feats])

    # apply one-hot encoding, remove 1 column to keep the minimum of columns
    data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

    # reformat data
    if 'label_deathrate' in data.columns:
        data['target'] = data['label_deathrate']
        data.drop(['label_deathrate'], axis=1, inplace=True)
    return data

train = feature_eng(train)
test = feature_eng(test)


# ------------ Step 4. Fitting the model ------------
x_train, y_train = split_feat_label(train)
x_test, y_test = split_feat_label(test)

# Replacing missing values with their respective means.
col_mean = x_train.mean()
x_train = x_train.fillna(col_mean)
x_test = x_test.fillna(col_mean)

# Here we shall test our model by doing a 10-fold cross-validation.
folds = KFold(n_splits=10, shuffle=True, random_state=seed)

# Number of features.
numfeatures = x_train.shape[1]

gridparameters = [{'n_features_to_select': list(range(2, numfeatures))}]

modelLR = LinearRegression()
rfeLR = RFE(modelLR)

modelrfeCV = GridSearchCV(estimator=rfeLR, param_grid=gridparameters, scoring='r2', cv=folds, verbose=1,
                          return_train_score=True)

# Fitting the model.
modelrfeCV.fit(x_train, y_train)

# What results are obtained by our cross-validation?
modelrfeCV_results = pd.DataFrame(modelrfeCV.cv_results_)

# Plotting those results.
plt.figure(figsize=(16, 6))

plt.plot(modelrfeCV_results["param_n_features_to_select"], modelrfeCV_results["mean_test_score"])
plt.plot(modelrfeCV_results["param_n_features_to_select"], modelrfeCV_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('R-squared')
plt.title("R-squared by number of features")
plt.legend(['validation score', 'train score'])
plt.grid(True)
plt.show();

# Looking at the graph, we can choose to keep 15 features in our final model as the slope seems to remain steady.
# We will choose a simpler model over one that contains more predictors as some noise could be generated behind the scenes.

# Final model
best_numfeatures = 15
LRfinal = LinearRegression()
LRfinal.fit(x_train, y_train)

rfeLRfinal = RFE(LRfinal, n_features_to_select=best_numfeatures)
rfeLRfinal.fit(x_train, y_train)

list(zip(df.iloc[:, :-1].columns, rfeLRfinal.support_, rfeLRfinal.ranking_))

# What is the R-squared obtained for the training set?
round(rfeLRfinal.score(x_train, y_train), 4)

# Let's measure our test error.
scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
results_final = cross_validate(rfeLRfinal, x_train, y_train, scoring=scoring, cv=folds, return_train_score=True)

results_finaldf = pd.DataFrame(results_final)
results_finaldf

# Calculating different metrics pertaining to the validity of our model during training: R2, MAE, MSE.
print("The following metrics were obtained for the training set: ")
print('   R-squared:', round(results_finaldf['train_r2'].mean(), 4))
print('   Standard deviation R-squared:', round(results_finaldf['train_r2'].std(), 4))
print('   Mean Absolute Error:', round(-1 * results_finaldf['train_neg_mean_absolute_error'].mean(), 4))
print('   Mean Squared Error:', round(-1 * results_finaldf['train_neg_mean_squared_error'].mean(), 4))

# Calculating different metrics pertaining to the validity of our model during validation: R2, MAE, MSE.
print("The following metrics were obtained for the validation set: ")
print('   R-squared:', round(results_finaldf['test_r2'].mean(), 4))
print('   Standard deviation R-squared:', round(results_finaldf['test_r2'].std(), 4))
print('   Mean Absolute Error:', round(-1 * results_finaldf['test_neg_mean_absolute_error'].mean(), 4))
print('   Mean Squared Error:', round(-1 * results_finaldf['test_neg_mean_squared_error'].mean(), 4))

# Let's visualize once more the results for our R-squared.
r2_testavg = [np.mean(results_finaldf['test_r2'])] * len(results_finaldf)
plt.figure(figsize=(16, 6))
plt.plot(results_finaldf.index, results_finaldf['train_r2'], label='R-squared train', marker='o', color='pink')
plt.plot(results_finaldf.index, results_finaldf['test_r2'], label='R-squared test', marker='o', color='blue')
plt.plot(results_finaldf.index, r2_testavg, label='Mean R-squared test', linestyle='--', color='red')
plt.xlabel('Iteration')
plt.ylabel('R-squared')
plt.title('R-squared per iteration')
plt.legend(loc='lower left')
plt.show;


# ------------ Step 5. Final Model Score ------------
# Our model is finally ready to be tested.
y_pred = rfeLRfinal.predict(x_test)
round(rfeLRfinal.score(x_test, y_test), 4)

print("Our final model has achieved the following performance on the Test Data:")
print('   Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred), 4))
print('   Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred), 4))
print('   Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 4))


# ------------ Step 6. Model Card Functions: Fetch, Train, Evaluate, Build_Paper ------------
# Download/create the dataset
def fetch():
    url = "https://raw.githubusercontent.com/MarthyGarcia/IFT6390_Project1/main/Data/cancer_reg.csv"
    data = pd.read_csv(url)
    return data

# Train your model on the dataset
def train():
    df = fetch()
    df['label_deathrate'] = df['target_deathrate']
    df.drop(['target_deathrate'], axis=1, inplace=True)
    
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Splitting the data: test, train 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, shuffle= True, random_state=seed)

    train = pd.concat([x_train, y_train], axis=1)
    test = pd.concat([x_test, y_test], axis=1)
    
    train = feature_eng(train)
    test = feature_eng(test)
    x_train, y_train = split_feat_label(train)
    x_test, y_test = split_feat_label(test)
    col_mean = x_train.mean()
    x_train = x_train.fillna(col_mean)
    x_test = x_test.fillna(col_mean)
    # Here we shall test our model by doing a 10-fold cross-validation.
    folds = KFold(n_splits = 10, shuffle = True, random_state = seed)

    # Number of features.
    numfeatures = x_train.shape[1]

    gridparameters = [{'n_features_to_select': list(range(2, numfeatures))}]

    modelLR = LinearRegression()
    rfeLR = RFE(modelLR)             

    modelrfeCV = GridSearchCV(estimator = rfeLR, 
                            param_grid = gridparameters, 
                            scoring= 'r2', 
                            cv = folds, 
                            verbose = 1,
                            return_train_score=True)     
    # Fitting the model.
    modelrfeCV.fit(x_train, y_train)
    
    best_numfeatures = 15 
    LRfinal = LinearRegression()
    LRfinal.fit(x_train, y_train)

    rfeLRfinal = RFE(LRfinal, n_features_to_select=best_numfeatures)
    rfeLRfinal.fit(x_train, y_train)
    model = rfeLRfinal
    
    return model, x_test, y_test

# Compute the evaluation metrics and figures
def evaluate():
    print("Training the model and calculating the Test R-squared...")
    model, x_test, y_test = train()
    return round(model.score(x_test, y_test), 4)

# Compile the PDF documents
def build_paper():
    import urllib.request
    download_url = 'https://github.com/MarthyGarcia/IFT6390_Project1/raw/main/Pdf/card.pdf'
    filename = "card"
    response = urllib.request.urlopen(download_url)    
    file = open(filename + ".pdf", 'wb')
    file.write(response.read())
    file.close()
    download_url = 'https://github.com/MarthyGarcia/IFT6390_Project1/raw/main/Pdf/paper.pdf'
    filename = "paper"
    response = urllib.request.urlopen(download_url)    
    file = open(filename + ".pdf", 'wb')
    file.write(response.read())
    file.close()


###############################
# No need to modify past here #
###############################

supported_functions = {'fetch': fetch,
                       'train': train,
                       'evaluate': evaluate,
                       'build_paper': build_paper}

# If there is no command-line argument, return an error
if len(sys.argv) < 2:
    print("""You need to pass in a command-line argument. Choose among 'fetch', 'train', 'evaluate' and 'build_paper'.""")
    sys.exit(1)

# Extract the first command-line argument, ignoring any others
arg = sys.argv[1]

# Run the corresponding function
if arg in supported_functions:
    supported_functions[arg]()
else:
    raise ValueError("""'{}' not among the allowed functions. Choose among 'fetch', 'train', 'evaluate' and 'build_paper'.
    """.format(arg))
