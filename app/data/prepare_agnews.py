import json
import zipfile
import re

from typing import Callable, Dict, List, Tuple
from pathlib import Path

import click
import gensim.downloader as gensim_downloader
import numpy as np
import requests

from gensim.utils import tokenize as gensim_tokenizer
from gensim.parsing.preprocessing import remove_stopwords
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


class TransformerFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, dim: int=768, model_name: str="sentence-transformers/all-mpnet-base-v2"):
        self._dim = dim
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(f"Transforming {len(X)} documents to word vectors based on Transformer model '{self._model_name}'...")
        return self._model.encode(
            sentences=X,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=256,
            device="cuda:0"
        )


class AverageWordVectorFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, dim: int=100, vector_model_name: str = "glove-wiki-gigaword-100"):
        self._dim = dim
        self._vector_model_name = vector_model_name
        print(f"Loading word vector model {vector_model_name}...")
        self._word_vector_model = gensim_downloader.load(vector_model_name)
        self._empty_doc_vector = np.zeros(self._dim)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        print(f"Transforming {len(X)} documents to average word vectors...")
        X_t = []
        with tqdm(total=len(X)) as pbar:
          for doc_idx, doc in enumerate(X):
              tokens = list(gensim_tokenizer(doc, lowercase=True, deacc=True))
              word_vectors_in_doc = [
                self._word_vector_model[token]
                for token in tokens
                if token in self._word_vector_model
              ]
              if len(word_vectors_in_doc) == 0:
                  X_t.append(self._empty_doc_vector)
              else:
                  X_t.append(np.mean(word_vectors_in_doc, axis=0))
              pbar.update()
        return np.array(X_t)


def get_data_splits_from_zip_file(zip_path: Path) -> None:
    data_splits = {}
    with zipfile.ZipFile(zip_path, "r") as z:
        for split in ["train", "test", "augment"]:
            print(f"Reading {split}.json from {zip_path}...")
            with z.open(f"{split}.json") as f:
                data_splits[split] = json.load(f)
    return data_splits


class DataPreparation:
    def __init__(self, download_url: str, output_dir: str,) -> None:
        self._download_url = download_url
        self._output_dir = Path(output_dir)

    def run(self) -> None:
        raw_path = self._output_dir / "raw" / "agnews.zip"
        if not raw_path.exists():
            self._download_file(self._download_url, raw_path)
        data_splits = get_data_splits_from_zip_file(raw_path)
        docs_map, labels_map = self._get_docs_and_labels(data_splits)

        self._process_features_and_labels(
            docs_map=docs_map,
            labels_map=labels_map,
            featurizer_name="avg-word-model",
            featurizer_init_fn=AverageWordVectorFeaturizer,
        )

        self._process_features_and_labels(
            docs_map=docs_map,
            labels_map=labels_map,
            featurizer_name="transformer-model",
            featurizer_init_fn=TransformerFeaturizer,
        )

    def _process_features_and_labels(
        self,
        docs_map: Dict[str, List[str]],
        labels_map: Dict[str, List[str]],
        featurizer_name: str,
        featurizer_init_fn: Callable[[], TransformerMixin],
    ) -> None:
        featurizer: TransformerMixin = None
        for split_name, docs in docs_map.items():
            print(f"Processing {split_name} data...")
            split_path = self._output_dir / "processed" / f"agnews-{split_name}-{featurizer_name}.npz"
            if split_path.exists():
                print(f"File '{split_path}' already exists, skipping...")
                continue
            if featurizer is None:
                featurizer = featurizer_init_fn()
            split_path.parent.mkdir(parents=True, exist_ok=True)
            labels = labels_map[split_name]
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(labels)
            features = featurizer.fit_transform(docs, labels)
            print(f"Storing Numpy data to {split_path}...")
            np.savez_compressed(split_path, features=features, labels=y, class_names=label_encoder.classes_)

    def _get_docs_and_labels(self, data_splits: dict) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Extract descriptions and corresponding labels for each data split.   """
        docs, labels = {}, {}
        for split_name in data_splits:
            docs[split_name] = []
            labels[split_name] = []
            for item in data_splits[split_name]:
                desc, label = item["description"].strip(), item["label"]
                if len(desc) > 0:
                    docs[split_name].append(desc)
                    labels[split_name].append(label)
                else:
                    print(f"Discarding item because description is empty for item\n {item}")
            print(f"Read {len(docs[split_name])} documents from the {split_name} split.")
        return docs, labels

    def _download_file(self, url: str, output_path: Path) -> None:
        """Downloads a file from a url and saves it to a local path."""
        print(f"Downloading {url}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True) as r:
            with open(output_path, "wb") as f:
                total_length = int(r.headers.get("content-length", 0))
                with tqdm.tqdm(total=total_length, unit="iB", unit_scale=True) as pbar:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))


@click.command(help="Prepares the AG News data set.")
@click.option(
    "-u",
    "--download-url",
    type=click.STRING,
    required=False,
    default="https://corise-mlops.s3.us-west-2.amazonaws.com/project1/agnews.zip",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.STRING,
    required=False,
    default="data/input",
)
def main(
    download_url: str,
    output_dir: str,
):
    DataPreparation(
        download_url=download_url,
        output_dir=output_dir,
    ).run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
