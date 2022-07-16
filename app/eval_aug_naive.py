from pathlib import Path

import click
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm


class EvaluationWithNaiveAugmentation:
    def __init__(self, train_data_path: str, test_data_path, augment_data_path: str) -> None:
        self._train_data_path = Path(train_data_path)
        self._test_data_path = Path(test_data_path)
        self._augment_data_path = Path(augment_data_path)

        for file_path in [self._train_data_path, self._test_data_path]:
            if not file_path.exists():
                raise ValueError(f"File '{file_path}' does not exist.")

    def run(self) -> None:
        print(f"Loading training data from '{self._train_data_path}'...")
        train_data = np.load(self._train_data_path)
        print(f"Loading test data from '{self._test_data_path}'...")
        test_data = np.load(self._test_data_path)
        print(f"Loading augment data from '{self._augment_data_path}'...")
        augment_data = np.load(self._augment_data_path)


        X_train, y_train = train_data["features"], train_data["labels"]
        X_test, y_test = test_data["features"], test_data["labels"]
        X_augment, y_augment = augment_data["features"], augment_data["labels"]
        class_names = train_data["class_names"]

        results = dict()

        for augment_size in [1000, 5000, 10000, 50000]:
            print(f"Evaluating when training is augmented with {augment_size} elements from the augment split.")

            X_train_i = np.concatenate((X_train, X_augment[:augment_size]), axis=0)
            y_train_i = np.concatenate((y_train, y_augment[:augment_size]), axis=0)

            print(f"Shape of training data: {X_train_i.shape}")

            ml_algo = LogisticRegression(max_iter=4000)
            ml_algo.fit(X_train_i, y_train_i)
            y_pred_i = ml_algo.predict(X_test)
            
            # record results
            results[augment_size] = {
                'accuracy': accuracy_score(y_test, y_pred_i),
                'f1': f1_score(y_test, y_pred_i, average='weighted'),
                'errors': sum([x != y for (x, y) in zip(y_test, y_pred_i)])
            }
            print(classification_report(y_test, y_pred_i, target_names=class_names))
        
        print(f"\nSummary of results:\n===================\n")
        for n_points, info in results.items():
            print(f" Augment size: {n_points:5d}  |  Accuracy: {info['accuracy']:0.4f}  |  F1 score: {info['f1']:0.4f}  |  Error count: {info['errors']}")
        

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
    "-e",
    "--augment-data-path",
    type=click.STRING,
    required=False,
    default="data/input/processed/agnews-augment-avg-word-model.npz",
)
def main(
    train_data_path: str,
    test_data_path: str,
    augment_data_path: str,
):
    EvaluationWithNaiveAugmentation(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        augment_data_path=augment_data_path,
    ).run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
