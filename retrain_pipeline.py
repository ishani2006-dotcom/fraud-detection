import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

path = 'AIML Dataset.csv'
df = pd.read_csv(path)
print('shape', df.shape)

cols_to_drop = ['nameOrig', 'nameDest']
if 'isFlaggedFraud' in df.columns:
    cols_to_drop.append('isFlaggedFraud')

df_model = df.drop(cols_to_drop, axis=1)
y = df_model['isFraud']
X = df_model.drop('isFraud', axis=1)

categorical = ['type']
numeric = ['amount', 'oldbalanceOrg', 'oldbalanceDest', 'newbalanceDest']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical)
    ],
    remainder='drop'
)
model = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])
model.fit(X_train, y_train)
print('train score', model.score(X_train, y_train))
print('test score', model.score(X_test, y_test))
joblib.dump(model, 'fraud_detection_pipeline.pkl')
print('saved model to fraud_detection_pipeline.pkl')

row = pd.DataFrame([{
    'type': 'DEPOSIT',
    'amount': 1000.0,
    'oldbalanceOrg': 1000.0,
    'oldbalanceDest': 0.0,
    'newbalanceDest': 0.0
}])
print('prediction for DEPOSIT:', model.predict(row)[0])
