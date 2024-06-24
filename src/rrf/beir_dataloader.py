import os
from typing import Any, Dict, Optional, Tuple

from beir import util
from beir.datasets.data_loader import GenericDataLoader


class BeirDataloader:

    def __init__(self, dataset: str):
        self.dataset = dataset

    def download_and_unzip(self):
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset}.zip"
        out_dir = os.path.join(os.getcwd(), "datasets")
        self.data_path = util.download_and_unzip(url, out_dir)
        print(f"Dataset downloaded here: {self.data_path}")
        return self.data_path

    def load(
        self, data_path: Optional[str] = None, split: str = "test"
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        if data_path:
            self.data_path = data_path
        corpus, queries, qrels = GenericDataLoader(self.data_path).load(split=split)
        return corpus, queries, qrels
