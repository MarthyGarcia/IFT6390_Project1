# Importing dependencies
import sys
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


# ------------ Model Card Functions: Fetch, Train, Evaluate, Build_Paper ------------
# Here we look to finalize cleaning the data and transforming the variables.
def feature_eng(data):
    # features to drop
    drop_features = ['pctsomecol18_24','geography']
    
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
    data = pd.get_dummies(data,columns =categorical_features, drop_first=True)

    # reformat data
    if 'label_deathrate' in data.columns:
        data['target'] = data['label_deathrate']
        data.drop(['label_deathrate'], axis=1, inplace=True)
    return data

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

    # Set seed
    seed = 1000
    
    # Splitting the data: test, train 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, shuffle= True, random_state=seed)

    train = pd.concat([x_train, y_train], axis=1)
    test = pd.concat([x_test, y_test], axis=1)
    
    train = feature_eng(train)
    test = feature_eng(test)
    x_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    x_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]
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
    return print(round(model.score(x_test, y_test), 4))

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
