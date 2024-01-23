import argparse
import os
import pickle
import subprocess
import sys
import time
import pika.exceptions
import threading
import json
import signal

from typing import Any
from enum import Enum

from fastgres.baseline.utility import chunks
from fastgres.async_components.node import SenderNode, ReceiverNode
from fastgres.baseline.log_utils import get_logger
from fastgres.query_encoding.query import Query


class DispatchMode(Enum):
    FORWARD = 1
    BATCH = 2


class DispatcherSender(SenderNode):

    def __init__(self, name: str, queues: list[str]):
        super().__init__(name, queues)

    def __str__(self):
        return f"Dispatcher Sending Node: {self.name}"


class DispatcherReceiver(ReceiverNode):

    def __init__(self, name: str, queue: str, mode: DispatchMode):
        super().__init__(name, [queue])
        print(f"[{self.name} *] Listening on queues: {self.queues}")
        self.dispatch_mode = mode
        self.batch_size = 1 if self.dispatch_mode == DispatchMode.FORWARD else 1000
        self.timeout = 1.0
        self.dispatched_nodes = dict()
        self.processes = list()
        self.model_path = os.path.join(os.path.dirname(__file__),
                                       "..",
                                       "models/saved_models/stack/fastgres.pkl")
        self.enc_path = os.path.join(os.path.dirname(__file__),
                                     "..",
                                     "database_statistics/stack/")
        self.sender = None
        self.rebuild_sender = True

    def __str__(self):
        return f"Query Dispatcher Receiver Node: {self.name}"

    def _queue_callback(self, ch: Any, method: Any, properties: Any, body: Any):
        try:
            decoded: dict = json.loads(body)
        except Exception as e:
            raise e
        message = decoded["message"]
        print(f"[{self.name} *] Received Payload: {message}")
        provider_query_prefix = "provider.dispatcher.query"
        if message.startswith(provider_query_prefix):
            # this message originates from a query provider and the payload contains a query
            self._handle_query_payload(decoded["payload"])
        else:
            print(f"[{self.name} *] Received unknown body: {message}")

    def _handle_query_payload(self, payload: dict):
        # print(f"[{self.name} *] Handling query names: {list(payload.keys())}")
        query_components = list(payload.items())
        if self.batch_size > 1:
            query_component_chunks = chunks(query_components, self.batch_size)
        else:
            query_component_chunks = query_components
        for component_chunk in query_component_chunks:
            self._handle_query_batch(component_chunk)

    def _handle_query_batch(self, chunk_list: list[tuple[str, str]]):
        # print(f"[{self.name} *] Handling query batch")
        thread_count = 0
        for query_name, query in chunk_list:
            thread = threading.Thread(target=self._handle_query,
                                      args=[query_name, query],
                                      daemon=True)
            thread.start()
            thread.join()
            thread_count += 1
        print(f"[{self.name} *] Used {thread_count} threads.")

    def _handle_query(self, query_name: str, query: str):
        parsed_query = Query(query_name, query)
        context = parsed_query.context
        # print(f"[{self.name} *] Handled Query: {parsed_query.name} with Context: {parsed_query.context}")

        # decide whether to open a new node or use an existing one to forward a query
        if context not in self.dispatched_nodes:
            node_name = f"Context_{len(self.dispatched_nodes.keys())}_Processing_Node"
            channel = f"context.{len(self.dispatched_nodes.keys())}.processing"

            # TODO: open new node for context
            # opening as module
            # python -m fastgres.async_components.query_dispatcher dispatcher -c query_queue
            venv_path = os.path.join(os.path.dirname(__file__),
                                     "..", "..",
                                     "venv/Scripts/python")
            process = subprocess.Popen(f"{venv_path} -m fastgres.async_components.context_handler "
                                       f"{node_name} "
                                       f"-c {channel} "
                                       f"-mp {self.model_path} "
                                       f"-ep {self.enc_path}")
            self.processes.append((process, node_name))
            self.dispatched_nodes[context] = (node_name, channel)
            self.rebuild_sender = True
        node_name, channel = self.dispatched_nodes[context]

        # now send the parsed query to the running processing node
        message = {"message": "dispatcher.processor.query",
                   "payload": [parsed_query]}

        if self.rebuild_sender:
            all_channels = [_[1] for _ in self.dispatched_nodes.values()]
            self.sender = DispatcherSender("Dispatch_Sender", all_channels)
            self.rebuild_sender = False
        self.sender.send_message(pickle.dumps(message), self.dispatched_nodes[context][1])


def main(name: str, queue: str, mode: DispatchMode):
    node = DispatcherReceiver(name, queue, mode)
    print(f'[{name} *] Waiting for query. To exit press CTRL+C')
    node.run()
    if node.sender:
        node.sender.close()

    for process, node_name in node.processes:
        print(f"Killing process node: {node_name}")
        process.send_signal(signal.CTRL_C_EVENT)
    print("Exited gracefully.")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Distributed Workload Evaluation using Fastgres.")
    parser.add_argument("name", default=None, help="Context Node Name.")
    parser.add_argument("-c", "--channel", default=None, help="Channel to listen to.")
    parser.add_argument("-dm", "--dispatchmode", default='batch', help="Mode to dispatch queries. Either 'forward' "
                                                                       "or 'batch' is valid.")
    args = parser.parse_args()

    if args.name is None:
        raise ValueError("Name parameter is None.")
    if args.channel is None:
        raise ValueError("Channel parameter is None.")
    if args.dispatchmode is None:
        raise ValueError("Dispatch Mode parameter is None.")
    if args.dispatchmode.lower() in 'forward':
        args_mode = DispatchMode.FORWARD
    elif args.dispatchmode.lower() in 'batch':
        args_mode = DispatchMode.BATCH
    else:
        raise ValueError("Dispatch Mode parameter not 'forward' or 'batch'.")

    main(str(args.name), str(args.channel), args_mode)
