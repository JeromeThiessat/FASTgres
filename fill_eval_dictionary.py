
import utility as u
import argparse

from hint_sets import HintSet
from tqdm import tqdm


def run():
    # done (c for complete):
    # represent - c
    # full_random - 10-c, 20-c, 30-c, 40-c, 50-c, 60-c, 70-c, 80-c, 90-c,
    # context_random - 10-c, 20-c, 30-c, 40-c, 50-, 60-, 70-, 80-, 90-,
    #
    # evaluation/context_random/10/fastgres/prediction.json
    # python fill_eval_dictionary.py -p evaluation/context_random/90/fastgres/prediction.json -s initial -b True evaluation/eval_dict.json
    # python fill_eval_dictionary.py -p evaluation/context_random/90/fastgres/prediction.json -s intermediate -b True evaluation/eval_dict.json
    # python fill_eval_dictionary.py -p evaluation/context_random/90/fastgres/prediction.json -s final -b True evaluation/eval_dict.json
    # python fill_eval_dictionary.py -p stack_test_prediction.json evaluation/eval_dict.json
    print('Using Dictionary Filling v.0.0.6 - Stack')
    parser = argparse.ArgumentParser(description="Evaluate Fastgres on given strategy")
    parser.add_argument("eval", default=None, help="Evaluation dictionary path")
    parser.add_argument("-p", "--prediction", default=None,
                        help="Path to the prediction dict to consider")
    parser.add_argument("-s", "--stage", default=None, help="Prediction stage to consider. "
                                                            "Choose from initial, intermediate, and final. "
                                                            "Defaults to None which indicates no stage distinction.")
    parser.add_argument("-b", "--bao", default=False,
                        help="Additionally evaluates five Hint Sets from BAO if not done already. "
                             "Can be set to True or False")
    parser.add_argument("-a", "--addition", default=None, help="Path to additional hints to consider (JSON-list)")
    parser.add_argument("-qp", "--querypath", default="stack", help="Queries to use. stack or job.")
    args = parser.parse_args()

    eval_path = args.eval
    prediction_path = args.prediction
    eval_bao = True if args.bao == "True" else False
    if args.addition is None:
        args_addition = list()
    else:
        args_addition = list(u.load_json(args.addition))

    eval_dict = u.load_json(eval_path)
    prediction_dict = u.load_json(prediction_path)

    if args.stage is None:
        stage_dict = prediction_dict
    else:
        stage_dict = prediction_dict[args.stage]
    queries = stage_dict.keys()
    args_query_path = args.querypath
    if args_query_path == "stack":
        q_path = "queries/stack/all/"
    elif args_query_path == "job":
        q_path = "queries/job/"
    else:
        raise ValueError("Query path input -qp -> {} not recognized".format(args_query_path))

    # possible BAO 'Arms' we take care of 63 is pg default and already done
    # 111 111 -> 63
    # 110 111 -> 63 - 8 = 55
    # 101 111 -> 63 - 16 = 47
    # 101 011 -> 47 - 4 = 43
    # 100 011 -> 43 - 8 = 35
    bao_hints = [55, 47, 43, 35]

    for query in tqdm(queries):
        pg_default_time = eval_dict[query]['63']
        # double default time with 1 second offset suffices to show bad queries, even for smaller loads
        timeout = int(2 * pg_default_time + 1)

        prediction = stage_dict[query]

        hints_to_evaluate = set(args_addition)
        try:
            _ = eval_dict[query][str(prediction)]
        except KeyError:
            hints_to_evaluate.add(int(prediction))

        if eval_bao:
            for bao_hint in bao_hints:
                try:
                    _ = eval_dict[query][str(bao_hint)]
                except KeyError:
                    hints_to_evaluate.add(int(bao_hint))

        for hint_set_int in hints_to_evaluate:

            try:
                # catch additions
                _ = eval_dict[query][str(hint_set_int)]
                continue
            except KeyError:
                pass

            eval_hint_set = HintSet(hint_set_int)
            pred_eval = u.evaluate_hinted_query(q_path, query, eval_hint_set, u.PG_STACK_OVERFLOW, timeout)

            if pred_eval is None:
                # unneeded cast to be safe
                pred_eval = int(timeout)
            eval_dict[query][str(hint_set_int)] = pred_eval

    u.save_json(eval_dict, eval_path)
    return


if __name__ == "__main__":
    run()
