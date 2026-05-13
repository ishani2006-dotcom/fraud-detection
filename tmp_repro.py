import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

path = 'C:/Users/DELL/vs code/.vscode/AIML Dataset.csv'
df = pd.read_csv(path)

df_model = df.drop(["nameOrig","nameDest","isFlaggedFraud"], axis=1)
y = df_model["isFraud"]
X = df_model.drop("isFraud", axis=1)

numeric = ["amount", "oldbalanceOrg", "oldbalanceDest", "newbalanceDest"]
categorical = ["type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(drop="first"), categorical)
    ],
    remainder="drop"
)
model = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('score', model.score(X_test, y_test))
print(classification_report(y_test, y_pred))
