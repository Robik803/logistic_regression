import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.feature_selection import chi2, SelectKBest
import joblib

hyperparams = {
	"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
	"penalty": ["l1", "l2", "elasticnet", "none"],
	"solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    "max_iter": [200, 250, 300, 500]
}

def display_info_dataset(df: pd.DataFrame):
        
    print(df.info())
    print(df.head())
    print(df.shape)
    print(df.describe())

df = pd.read_csv("data/raw/bank-marketing-campaign-data.csv", delimiter=";")
display_info_dataset(dataframe=df)

df = df.drop_duplicates().reset_index(drop=True)
    
df["job_n"] = pd.factorize(df["job"])[0]
df["marital_n"] = pd.factorize(df["marital"])[0]
df["education_n"] = pd.factorize(df["education"])[0]
df["default_n"] = pd.factorize(df["default"])[0]
df["housing_n"] = pd.factorize(df["housing"])[0]
df["loan_n"] = pd.factorize(df["loan"])[0]
df["contact_n"] = pd.factorize(df["contact"])[0]
df["month_n"] = pd.factorize(df["month"])[0]
df["day_of_week_n"] = pd.factorize(df["day_of_week"])[0]
df["poutcome_n"] = pd.factorize(df["poutcome"])[0]
df["y_n"] = pd.factorize(df['y'])[0]

numeric_features = df.select_dtypes(include=["int64", "float64"]).columns
    
scaler = MinMaxScaler()
scal_features = scaler.fit_transform(df[numeric_features])
df = pd.DataFrame(scal_features, index = df.index, columns = numeric_features)
    
display_info_dataset(dataframe=df)

X = df.drop("y_n", axis=1)
y = df["y_n"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

model = SelectKBest(chi2, k=5)
model.fit(X_train, y_train)
features_selected = model.get_support()
X_train = pd.DataFrame(model.transform(X_train), columns = X_train.columns.values[features_selected])
X_test = pd.DataFrame(model.transform(X_test), columns = X_test.columns.values[features_selected])

display_info_dataset(X_train)
display_info_dataset(X_test)

X_train["y_n"] = list(y_train)
X_test["y_n"] = list(y_test)
X_train.to_csv("./data/processed/bank_data_train.csv", index = False)
X_test.to_csv("./data/processed/bank_data_test.csv", index = False)

train_data = pd.read_csv("./data/processed/bank_data_train.csv")
test_data = pd.read_csv("./data/processed/bank_data_test.csv")

X_train = train_data.drop("y_n", axis=1)
y_train = train_data["y_n"]
X_test = test_data.drop("y_n", axis=1)
y_test = test_data["y_n"]

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"ROC-AUC score: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])}")

grid = GridSearchCV(model, hyperparams, scoring = "recall", cv = 5)
grid.fit(X_train, y_train)
print(f"Best hyperparamters: {grid.best_params_}")

model = LogisticRegression(C = 1000, max_iter = 200, penalty = 'l2', solver = 'lbfgs')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"ROC-AUC score: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])}")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

joblib.dump(model, "./data/processed/bank_model.pkl")
