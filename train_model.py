import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load the dataset
df = pd.read_csv('final-v1.csv')

# Select relevant features and target
features = [
    'username_length',
    'username_has_number',
    'full_name_has_number',
    'full_name_length',
    'is_private',
    'is_joined_recently',
    'has_channel',
    'is_business_account',
    'has_guides',
    'has_external_url',
    'edge_followed_by',
    'edge_follow'
]
target = 'is_fake'

X = df[features]
y = df[target]

# Convert boolean True/False to 1/0
X = X.replace({True: 1, False: 0})
y = y.replace({True: 1, False: 0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model/scam_model.pkl')

# Evaluate model
y_pred = model.predict(X_test)
print("Model Evaluation:\n")
print(classification_report(y_test, y_pred))

