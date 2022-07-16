from pathlib import Path

import click
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm


class EvaluationWithAllTrainingData:
    def __init__(self, train_data_path: str, test_data_path) -> None:
        self._train_data_path = Path(train_data_path)
        self._test_data_path = Path(test_data_path)

        for file_path in [self._train_data_path, self._test_data_path]:
            if not file_path.exists():
                raise ValueError(f"File '{file_path}' does not exist.")

    def run(self) -> None:
        print(f"Loading training data from '{self._train_data_path}'...")
        train_data = np.load(self._train_data_path)
        print(f"Loading test data from '{self._test_data_path}'...")
        test_data = np.load(self._test_data_path)

        X_train, y_train = train_data["features"], train_data["labels"]
        X_test, y_test = test_data["features"], test_data["labels"]
        class_names = train_data["class_names"]

        print("Training model...")
        ml_algo = LogisticRegression(max_iter=4000)
        ml_algo.fit(X_train, y_train)
        
        print("Generating predictions...")
        y_pred = ml_algo.predict(X_test)

        error_count = sum([x != y for (x, y) in zip(y_test, y_pred)])
        print(f"Error count: {error_count}")

        print(classification_report(y_test, y_pred, target_names=class_names))


@click.command(help="Evaluation.")
@click.option(
    "-t",
    "--train-data-path",
    type=click.STRING,
    required=False,
    default="data/input/processed/agnews-train-avg-word-model.npz",
)
@click.option(
    "-e",
    "--test-data-path",
    type=click.STRING,
    required=False,
    default="data/input/processed/agnews-test-avg-word-model.npz",
)
def main(
    train_data_path: str,
    test_data_path: str,
):
    EvaluationWithAllTrainingData(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
    ).run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
