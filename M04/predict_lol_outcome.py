import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import pickle
import warnings

warnings.filterwarnings("ignore")
plt.style.use("ggplot")
random_state = 1
np.random.seed(random_state)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_info_columns', 200)

# Load Data
file_path = 'C:\\Users\\Ali\\Desktop\\Ali_Hajjar_MSD62A_RDI\\M04\\match_data_v5.csv'
new_data_path = 'C:\\Users\\Ali\\Desktop\\Ali_Hajjar_MSD62A_RDI\\M04\\new_data.csv'

column_names = [
    'matchId', 'blueTeamControlWardsPlaced', 'blueTeamWardsPlaced', 'blueTeamTotalKills', 'blueTeamDragonKills',
    'blueTeamHeraldKills', 'blueTeamTowersDestroyed', 'blueTeamInhibitorsDestroyed', 'blueTeamTurretPlatesDestroyed',
    'blueTeamFirstBlood', 'blueTeamMinionsKilled', 'blueTeamJungleMinions', 'blueTeamTotalGold', 'blueTeamXp',
    'blueTeamTotalDamageToChamps', 'redTeamControlWardsPlaced', 'redTeamWardsPlaced', 'redTeamTotalKills',
    'redTeamDragonKills', 'redTeamHeraldKills', 'redTeamTowersDestroyed', 'redTeamInhibitorsDestroyed',
    'redTeamTurretPlatesDestroyed', 'redTeamMinionsKilled', 'redTeamJungleMinions', 'redTeamTotalGold', 'redTeamXp',
    'redTeamTotalDamageToChamps', 'blueWin', 'Unnamed'
]

df = pd.read_csv(file_path, header=None, names=column_names)

# Convert necessary columns to numeric
numeric_columns = [
    'blueTeamControlWardsPlaced', 'blueTeamWardsPlaced', 'blueTeamTotalKills', 'blueTeamDragonKills',
    'blueTeamHeraldKills', 'blueTeamTowersDestroyed', 'blueTeamInhibitorsDestroyed', 'blueTeamTurretPlatesDestroyed',
    'blueTeamFirstBlood', 'blueTeamMinionsKilled', 'blueTeamJungleMinions', 'blueTeamTotalGold', 'blueTeamXp',
    'blueTeamTotalDamageToChamps', 'redTeamControlWardsPlaced', 'redTeamWardsPlaced', 'redTeamTotalKills',
    'redTeamDragonKills', 'redTeamHeraldKills', 'redTeamTowersDestroyed', 'redTeamInhibitorsDestroyed',
    'redTeamTurretPlatesDestroyed', 'redTeamMinionsKilled', 'redTeamJungleMinions', 'redTeamTotalGold', 'redTeamXp',
    'redTeamTotalDamageToChamps', 'blueWin'
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Basic Data Information
print(df.shape)
print(df.head(2))

# Remove useless columns
df = df.drop(columns=['Unnamed', 'matchId'])

# Impute missing values
imputer = SimpleImputer(strategy='mean')
df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# Ensure blueWin is binary and convert to integer
df['blueWin'] = df['blueWin'].astype(int)

# Create new features
df['diffMinionsKilled'] = df['blueTeamMinionsKilled'] - df['redTeamMinionsKilled']
df['diffJungleMinions'] = df['blueTeamJungleMinions'] - df['redTeamJungleMinions']
df['diffTotalGold'] = df['blueTeamTotalGold'] - df['redTeamTotalGold']
df['diffTotalKills'] = df['blueTeamTotalKills'] - df['redTeamTotalKills']
df['diffXp'] = df['blueTeamXp'] - df['redTeamXp']
df['diffTotalDamageToChamps'] = df['blueTeamTotalDamageToChamps'] - df['redTeamTotalDamageToChamps']
df['diffDragonKills'] = df['blueTeamDragonKills'] - df['redTeamDragonKills']
df['diffHeraldKills'] = df['blueTeamHeraldKills'] - df['redTeamHeraldKills']
df['diffTowersDestroyed'] = df['blueTeamTowersDestroyed'] - df['redTeamTowersDestroyed']
df['diffInhibitorsDestroyed'] = df['blueTeamInhibitorsDestroyed'] - df['redTeamInhibitorsDestroyed']
df['diffTurretPlatesDestroyed'] = df['blueTeamTurretPlatesDestroyed'] - df['redTeamTurretPlatesDestroyed']
df.info()

# Data Split
feature_label = [
    'blueTeamControlWardsPlaced', 'blueTeamWardsPlaced', 'blueTeamTotalKills', 'blueTeamDragonKills',
    'blueTeamHeraldKills', 'blueTeamTowersDestroyed', 'blueTeamInhibitorsDestroyed', 'blueTeamTurretPlatesDestroyed',
    'blueTeamFirstBlood', 'blueTeamMinionsKilled', 'blueTeamJungleMinions', 'blueTeamTotalGold', 'blueTeamXp',
    'blueTeamTotalDamageToChamps', 'redTeamControlWardsPlaced', 'redTeamWardsPlaced', 'redTeamTotalKills',
    'redTeamDragonKills', 'redTeamHeraldKills', 'redTeamTowersDestroyed', 'redTeamInhibitorsDestroyed',
    'redTeamTurretPlatesDestroyed', 'redTeamMinionsKilled', 'redTeamJungleMinions', 'redTeamTotalGold', 'redTeamXp',
    'redTeamTotalDamageToChamps', 'diffMinionsKilled', 'diffJungleMinions', 'diffTotalGold', 'diffTotalKills', 'diffXp',
    'diffTotalDamageToChamps', 'diffDragonKills', 'diffHeraldKills', 'diffTowersDestroyed', 'diffInhibitorsDestroyed',
    'diffTurretPlatesDestroyed'
]

X = df[feature_label]
y = df['blueWin']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
scaler = preprocessing.RobustScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Size of training set: {}".format(len(X_train_scaled)))
print("Size of test set: {}".format(len(X_test_scaled)))

def find_bestKfeatures(model, score_func=f_classif):
    k = -1
    max_score = 0
    for i in range(1, len(feature_label)):
        selector = SelectKBest(score_func=score_func, k=i)
        pipeline = Pipeline([('selector', selector), ('model', model)])
        pipeline.fit(X_train_scaled, y_train)
        score = pipeline.score(X_test_scaled, y_test)
        print("K: {}, score: {}".format(i, score))
        if score > max_score:
            k = i
            max_score = score
            selected_features_indices = selector.get_support(indices=True)
    print("The best K number: {}, score: {}".format(k, max_score))
    return list(selected_features_indices)

def train_model_with_random_search(model, param_grid, X_train, y_train, X_test, y_test, skf):
    random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10,
                                       scoring='accuracy', n_jobs=-1, cv=skf)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    best_param = random_search.best_params_
    print('Best Parameters: ', best_param)
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred)

    print('Accuracy: ', accuracy)
    print('F1 Score: ', f1)
    print('AUC(ROC): ', roc_auc)
    print("Classification Report: ")
    print(classification_report(y_test, y_pred))


    return best_model, accuracy, f1, roc_auc, best_param

# Logistic Regression
log_reg_param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'liblinear'],
    'max_iter': [100, 200, 300, 400, 500]
}

model_lr = LogisticRegression()
lr_features = find_bestKfeatures(model_lr, score_func=f_classif)
X_train_lr, X_test_lr = X_train_scaled[:, lr_features], X_test_scaled[:, lr_features]
cv = StratifiedKFold(n_splits=5, shuffle=True)
model_lr, acc_lr, f1_lr, roc_auc_lr, param_lr = train_model_with_random_search(model_lr, log_reg_param_grid, X_train_lr, y_train, X_test_lr, y_test, cv)

# Save the model
with open('model_lr.pkl', 'wb') as file:
    pickle.dump(model_lr, file)

# Load new data
new_df = pd.read_csv(new_data_path, header=None, names=column_names)

# Ensure necessary columns are numeric
for col in numeric_columns:
    new_df[col] = pd.to_numeric(new_df[col], errors='coerce')

# Remove useless columns
new_df = new_df.drop(columns=['Unnamed', 'matchId'])

# Impute missing values in new data
new_df[numeric_columns] = imputer.transform(new_df[numeric_columns])

# Create new features for the new data
new_df['diffMinionsKilled'] = new_df['blueTeamMinionsKilled'] - new_df['redTeamMinionsKilled']
new_df['diffJungleMinions'] = new_df['blueTeamJungleMinions'] - new_df['redTeamJungleMinions']
new_df['diffTotalGold'] = new_df['blueTeamTotalGold'] - new_df['redTeamTotalGold']
new_df['diffTotalKills'] = new_df['blueTeamTotalKills'] - new_df['redTeamTotalKills']
new_df['diffXp'] = new_df['blueTeamXp'] - new_df['redTeamXp']
new_df['diffTotalDamageToChamps'] = new_df['blueTeamTotalDamageToChamps'] - new_df['redTeamTotalDamageToChamps']
new_df['diffDragonKills'] = new_df['blueTeamDragonKills'] - new_df['redTeamDragonKills']
new_df['diffHeraldKills'] = new_df['blueTeamHeraldKills'] - new_df['redTeamHeraldKills']
new_df['diffTowersDestroyed'] = new_df['blueTeamTowersDestroyed'] - new_df['redTeamTowersDestroyed']
new_df['diffInhibitorsDestroyed'] = new_df['blueTeamInhibitorsDestroyed'] - new_df['redTeamInhibitorsDestroyed']
new_df['diffTurretPlatesDestroyed'] = new_df['blueTeamTurretPlatesDestroyed'] - new_df['redTeamTurretPlatesDestroyed']

# Select features for the new data
new_X = new_df[feature_label]
new_X_scaled = scaler.transform(new_X)

# Select only the features used for training
new_X_selected = new_X_scaled[:, lr_features]

# Make prediction
prediction = model_lr.predict(new_X_selected)
probability = model_lr.predict_proba(new_X_selected)

print("Prediction:", prediction[1])
print("Probability of winning BL BW:", probability[1])

print("Prediction:", prediction[2])
print("Probability of winning BL BW:", probability[2])

print("Prediction:", prediction[3])
print("Probability of winning BL BW:", probability[3])

# ROC AUC for test data
y_pred_proba = model_lr.predict_proba(X_test_lr)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print('AUC(ROC): ', roc_auc)

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc_value = auc(fpr, tpr)

#plt.figure()
#plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_value)
#plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic (ROC) Curve')
#plt.legend(loc="lower right")
#plt.show()
