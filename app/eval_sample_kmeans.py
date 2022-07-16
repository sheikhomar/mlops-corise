import enum
from pathlib import Path

import click
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from sklearn.cluster import KMeans
from sklearn.utils.validation import check_array
from tqdm import tqdm


class EvaluationWithKMeansSampling:
    def __init__(self, train_data_path: str, test_data_path, target_size: int) -> None:
        self._train_data_path = Path(train_data_path)
        self._test_data_path = Path(test_data_path)
        self._target_size = target_size

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

        X_train_new = []
        y_train_new = []

        for i, class_name in enumerate(class_names):
            n_points_in_class = y_train[y_train == i].shape[0]
            if n_points_in_class > self._target_size:
                print(f"Class '{class_name}' is too large. Sampling {self._target_size} points from the class.")
                X_train_i = self._sample_points_via_kmeans_plus_plus(X_train[y_train == i], self._target_size)
                sampled_size = X_train_i.shape[0]
                X_train_new.append(X_train_i)
                y_train_new.append(y_train[y_train == i][:sampled_size])
            else:
                X_train_new.append(X_train[y_train == i])
                y_train_new.append(y_train[y_train == i])

        X_train_new = np.concatenate(X_train_new, axis=0)
        y_train_new = np.concatenate(y_train_new, axis=0)
        print(f"New training data: {X_train_new.shape[0]}")

        ml_algo = LogisticRegression(max_iter=4000)
        ml_algo.fit(X_train_new, y_train_new)
        y_pred = ml_algo.predict(X_test)

        error_count = sum([x != y for (x, y) in zip(y_test, y_pred)])
        print(f"Error count: {error_count}")

        print(classification_report(y_test, y_pred, target_names=class_names))

            
    def _sample_points_via_kmeans_plus_plus(self, X: np.ndarray, n_points: int) -> np.ndarray:
        kmeans = KMeans(n_clusters=n_points, init="k-means++", max_iter=1, n_init=1)
        kmeans.fit(X)
        return kmeans.cluster_centers_


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
@click.option(
    "-s",
    "--target-size",
    type=click.INT,
    required=False,
    default=300,
)
def main(
    train_data_path: str,
    test_data_path: str,
    target_size: int,
):
    EvaluationWithKMeansSampling(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        target_size=target_size,
    ).run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
