import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the data
print("Loading data...")
df = pd.read_csv(r"C:\Users\hp\Desktop\Jupyter_note_books_data_science\customer_churn_dataset.csv", encoding='latin1')

print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Encode categorical variables
print("\nEncoding categorical variables...")
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['SubscriptionType'] = le.fit_transform(df['SubscriptionType'])
df['PaymentMethod'] = le.fit_transform(df['PaymentMethod'])
df['ContractType'] = le.fit_transform(df['ContractType'])
df['HasDependents'] = le.fit_transform(df['HasDependents'])
df['Churn'] = le.fit_transform(df['Churn'])

# Prepare features and target
x = df.drop(columns=['Churn', 'CustomerID'])
y = df['Churn']

print(f"\nFeatures shape: {x.shape}")
print(f"Target shape: {y.shape}")

# Split the data
print("\nSplitting data...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=47)

# Train the Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=47, n_jobs=-1)
model.fit(x_train, y_train)

# Evaluate the model
train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)

print(f"\nModel trained successfully!")
print(f"Train accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")

# Save the model
print("\nSaving model...")
with open('churn_Random_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as 'churn_Random_model.pkl'")

# Test the model
print("\nTesting model with sample prediction...")
sample = x_test.iloc[0:1]
prediction = model.predict(sample)
probability = model.predict_proba(sample)
print(f"Sample prediction: {prediction[0]}")
print(f"Prediction probability: {probability[0]}")
