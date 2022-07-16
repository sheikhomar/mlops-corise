from pathlib import Path

import click
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


class Evaluation:
    def __init__(self, train_data_path: str, test_data_path) -> None:
        self._train_data_path = Path(train_data_path)
        self._test_data_path = Path(test_data_path)

        for file_path in [self._train_data_path, self._test_data_path]:
            if not file_path.exists():
                raise ValueError(f"File '{file_path}' does not exist.")

    def run(self) -> None:
        # Load numpy arrays from file
        print(f"Loading training data from '{self._train_data_path}'...")
        train_data = np.load(self._train_data_path)
        print(f"Loading test data from '{self._test_data_path}'...")
        test_data = np.load(self._test_data_path)

        X_train, y_train = train_data["features"], train_data["labels"]
        X_test, y_test = test_data["features"], test_data["labels"]

        results = dict()

        for train_size in [500, 1000, 2000, 5000, 10000, 25000]:
            print(f"Evaluating for training data size = {train_size}")

            X_train_i = X_train[:train_size]
            Y_train_i = y_train[:train_size]

            ml_algo = LogisticRegression(max_iter=4000)
            ml_algo.fit(X_train_i, Y_train_i)
            Y_pred_i = ml_algo.predict(X_test)
            
            # record results
            results[train_size] = {
                # 'test_predictions': Y_pred_i,
                'accuracy': accuracy_score(y_test, Y_pred_i),
                'f1': f1_score(y_test, Y_pred_i, average='weighted'),
                'errors': sum([x != y for (x, y) in zip(y_test, Y_pred_i)])
            }
            print(f"  Accuracy on test set: {results[train_size]['accuracy']}")
        

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
    Evaluation(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
    ).run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
