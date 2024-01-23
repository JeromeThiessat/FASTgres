
import argparse
import os
import pickle
import subprocess
import sys
import time
import threading
import json
import signal

from typing import Any
from enum import Enum

# preliminary pickle loading fix
from fastgres import query_encoding

sys.modules["query_encoding"] = query_encoding

from fastgres.baseline.utility import chunks
from fastgres.async_components.node import SenderNode, ReceiverNode
from fastgres.baseline.log_utils import get_logger
from fastgres.query_encoding.query import Query
from fastgres.baseline.utility import load_pickle, load_json
from fastgres.query_encoding.encoded_query import EncodedQuery
from fastgres.query_encoding.feature_extractor import EncodingInformation
from fastgres.models.sync_model import Synchronizer
from fastgres.query_encoding.encoded_query import Context


class ProcessingMode(Enum):
    FORWARD = 1
    BATCH = 2


class ProcessingReceiver(ReceiverNode):

    def __init__(self, name: str, queue: str, mode: ProcessingMode, model: Any, enc_path: str):
        super().__init__(name, [queue])
        self.dispatch_mode = mode
        self.batch_size = 1 if self.dispatch_mode == ProcessingMode.FORWARD else 2
        self.timeout = 1.0
        self.dispatched_nodes = dict()
        self.processes = list()

        # full model upon init, reduces once the first queries are recieved
        self.model = model
        self.reduced = False

        # Encoding Information, this only works if every path can be loaded
        self.encoding_info = EncodingInformation(None, enc_path, None)
        self.encoding_info.load_encoding_info()

    def __str__(self):
        return f"Query Processing Receiver Node: {self.name}"

    def _queue_callback(self, ch: Any, method: Any, properties: Any, body: Any):
        try:
            decoded: dict = pickle.loads(body)
        except Exception as e:
            raise e
        message = decoded["message"]
        # print(f"[{self.name} *] Received Payload: {message}")
        provider_query_prefix = "dispatcher.processor.query"
        if message.startswith(provider_query_prefix):
            # this message originates from a query dispatcher and the payload contains a query class
            self._handle_query_payload(decoded["payload"])
        else:
            print(f"[{self.name} *] Query Dispatcher Receiver received unknown body: {message}")

    def _handle_query_payload(self, payload: list[Query]):
        # print(f"[{self.name} *] Handling query names: {[_.name for _ in payload]}")
        if self.batch_size > 1:
            query_chunks = chunks(payload, self.batch_size)
        else:
            query_chunks = payload
        for component_chunk in query_chunks:
            self._handle_query_batch(component_chunk)

    def _handle_query_batch(self, chunk_list: list[Query]):
        # print(f"[{self.name} *] Handling query batch")
        for parsed_query in chunk_list:
            # first possibility to reduce the learned model towards its context
            # TODO: Handle model loading differently to avoid initial memory overhead
            if not self.reduced:
                reduced_model = self.model[parsed_query.context]
                del self.model
                self.model = reduced_model
                # self.model = self.model[parsed_query.context]
                self.reduced = True

            thread = threading.Thread(target=self._handle_query,
                                      args=[parsed_query],
                                      daemon=True)
            thread.start()
            thread.join()

    def _handle_query(self, parsed_query: Query):
        context_class: Context = Context()
        context_class.add_context(parsed_query.context)
        encoded_query = EncodedQuery(context_class, parsed_query, self.encoding_info)
        encoded_query.encode_query()
        synchronizer: Synchronizer = self.model
        prediction = synchronizer.model.predict(encoded_query.encoded_query)
        print(f"[{self.name} *] Predicted query: {parsed_query.name} with Hint Set: {prediction}")


def main(name: str, queue: str, mode: ProcessingMode, model_path: str, enc_path: str):
    full_model = load_pickle(model_path)
    print(f"[{name} *] Processor using queue: {queue}")
    node = ProcessingReceiver(name, queue, mode, full_model, enc_path)
    print(f'[{name} *] Waiting for query. To exit press CTRL+C')
    node.run()
    for process, node_name in node.processes:
        print(f"Killing execution node: {node_name}")
        process.send_signal(signal.CTRL_C_EVENT)
    print("Exited gracefully.")
    return


def check_encoding_paths(path: str):
    if not (os.path.exists(path + "mm_dict.pkl") and
            os.path.exists(path + "label_encoders.pkl") and
            os.path.exists(path + "wildcard_dict.json") and
            os.path.exists(path + "db_type_dict.json") and
            os.path.exists(path + "mm_dict.pkl")):
        return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Distributed Workload Evaluation using Fastgres.")
    parser.add_argument("name", default=None, help="Context Node Name.")
    parser.add_argument("-c", "--channel", default=None, help="Channel to listen to.")
    parser.add_argument("-dm", "--processmode", default='batch', help="Mode to dispatch queries. Either 'forward' "
                                                                      "or 'batch' is valid.")
    parser.add_argument("-mp", "--modelpath", default=None, help="Path to stored trained model.")
    parser.add_argument("-ep", "--encpath", default=None, help="Path to stored encoding dir.")
    args = parser.parse_args()

    if args.name is None:
        raise ValueError("Name parameter is None.")
    if args.channel is None:
        raise ValueError("Channel parameter is None.")
    if args.processmode is None:
        raise ValueError("Processing Mode parameter is None.")
    if args.processmode.lower() in 'forward':
        args_mode = ProcessingMode.FORWARD
    elif args.processmode.lower() in 'batch':
        args_mode = ProcessingMode.BATCH
    else:
        raise ValueError("Dispatch Mode parameter not 'forward' or 'batch'.")
    if args.modelpath is None or not os.path.exists(args.modelpath):
        raise ValueError(f"Model loading path: {args.modelpath} does not exist.")
    if args.encpath is None or not os.path.exists(args.encpath):
        raise ValueError(f"Model loading path: {args.encpath} does not exist.")
    # check if each path exists to avoid issues later on
    if not check_encoding_paths(args.encpath):
        raise ValueError("Encoding Information must be built beforehand")

    main(str(args.name), str(args.channel), args_mode, args.modelpath, args.encpath)
