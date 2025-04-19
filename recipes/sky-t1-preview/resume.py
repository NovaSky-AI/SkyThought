

from ray.data import Dataset
import ray 
from typing import Literal, List
import os 

def exclude_saved_entries(dataset: Dataset, save_dir: str, format: Literal["json", "parquet"], id_column: str):
    if not os.path.exists(save_dir) or not len(os.listdir(save_dir)):
        return dataset
    read_func = ray.data.read_json if format == "json" else ray.data.read_parquet
    saved_dataset = read_func(save_dir)
    saved_dataset_ids = set(saved_dataset.map(lambda x: {id_column: x[id_column]}).materialize().to_pandas()[id_column])
    dataset = dataset.filter(lambda x: x[id_column] not in saved_dataset_ids)
    return dataset

def resume_from_save_dir(datasets: List[Dataset], save_paths: List[str], format: Literal["json", "parquet"], id_column: str) -> List[Dataset]:
    for i in range(len(datasets)):
        datasets[i] = exclude_saved_entries(datasets[i], save_paths[i], format, id_column)
    return datasets


if __name__ == "__main__":
    ds = ray.data.from_items([{"id": i} for i in range(10)])
    import tempfile 
    with tempfile.TemporaryDirectory() as temp_dir:
        ds.filter(lambda x: x["id"] < 5).write_json(temp_dir)
        ds = exclude_saved_entries(ds, temp_dir, "json", "id")
        print(ds.take(10))
