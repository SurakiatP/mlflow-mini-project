import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

REGISTER_MODEL_NAME = "iris_rf_model"

def train_and_log():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("iris-classifier")

    with mlflow.start_run(run_name="rf-iris-run") as run:
        run_id = run.info.run_id
        model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 3)
        mlflow.log_metric("accuracy", acc)

        # Log model + auto register
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTER_MODEL_NAME
        )

        print(f" Model logged to run_id: {run_id}, Accuracy: {acc:.4f}")
        return run_id

def promote_latest_version():
    client = MlflowClient()
    latest_version = client.get_latest_versions(REGISTER_MODEL_NAME, stages=["None"])[-1].version

    # Promote latest version
    client.transition_model_version_stage(
        name=REGISTER_MODEL_NAME,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f" Promoted model version {latest_version} to Production")

if __name__ == "__main__":
    run_id = train_and_log()
    promote_latest_version()
