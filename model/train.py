from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("http://localhost:5000")
# Start MLflow experiment
mlflow.set_experiment("customer-churn-prediction")

# Load dataset
df = pd.read_csv("data/churn.csv")

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Convert categorical columns
X = pd.get_dummies(X)

# Save feature columns
joblib.dump(X.columns.tolist(), "model/features.pkl")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# MLflow tracking
with mlflow.start_run():

    # Model parameters
    n_estimators = 100
    random_state = 42

    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Log params
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Save model
    joblib.dump(model, "model/model.pkl")

    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Accuracy: {accuracy}")
    print("Model trained successfully!") 

    