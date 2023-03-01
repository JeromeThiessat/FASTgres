
import utility as u
from query import Query
from tqdm import tqdm, trange
from time import time


def main():
    print("Building Query Objects v.01")
    # adapt pahs if needed
    path = "queries/stack/all/"
    # path = "queries/job/"
    stack_queries = u.get_queries(path)
    save_path = "evaluation/stack/query_objects.pkl"
    save_path_time = "evaluation/stack/query_objects_encoding_time.json"
    # save_path = "evaluation/job/query_objects.pkl"
    # save_path_time = "evaluation/job/query_objects_encoding_time.json"

    query_objects = dict()
    encoding_time = dict()
    for query_name in tqdm(stack_queries):
        t0 = time()
        query_objects[query_name] = Query(query_name, path)
        encoding_time[query_name] = time() - t0

    u.save_pickle(query_objects, save_path)
    u.save_json(encoding_time, save_path_time)
    print("Done")

    return


if __name__ == "__main__":
    main()
