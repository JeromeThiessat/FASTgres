import json
import pickle
import numpy as np
import joblib

from typing import Any, Optional
from fastgres.baseline.hint_sets import HintSet


def load_json(path: str) -> Any:
    with open(path, 'r') as file:
        loaded = json.load(file)
    return loaded


def save_json(to_save: Any, path: str) -> None:
    json_dict = json.dumps(to_save)
    with open(path, 'w') as f:
        f.write(json_dict)
    return


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as file:
        loaded = pickle.load(file)
    return loaded


def save_pickle(to_save: Any, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(to_save, f)
    return


def load_joblib(path: str) -> Any:
    return joblib.load(path)


def save_joblib(to_save: Any, path: str) -> None:
    joblib.dump(to_save, path)
    return


def binary_to_int(bin_list: list[int]) -> int:
    return int("".join(str(x) for x in bin_list), 2)


def int_to_binary(integer: int, bin_size: int) -> list[int]:
    return [int(i) for i in bin(integer)[2:].zfill(bin_size)]


# def one_hot_to_binary(one_hot_vector: list[int]) -> list[int]:
#     ind = int(np.argmax(one_hot_vector))
#     return int_to_binary(ind)


# https://stackoverflow.com/a/312464
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def list_xor(a: list[int], b: list[int]):
    if len(a) != len(b):
        raise ValueError("Comparing two non equal length lists")
    return [0 if a[i] == b[i] else 1 for i in range(len(a))]


def get_first_mismatch(a: list[int], b: list[int]) -> Optional[int]:
    xor_result = list_xor(a, b)
    try:
        return xor_result.index(1)
    except ValueError:
        return None


def zip_and_order(a: list, b: list, order_by: int = 1, desc: bool = True) -> list[tuple]:
    sorted_list = sorted(zip(a, b), key=lambda x: x[order_by], reverse=desc)
    return sorted_list
