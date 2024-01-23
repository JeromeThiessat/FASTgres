
import argparse
import os
import subprocess
import sys
import threading
import pika

from fastgres.definitions import PathConfig
from fastgres.query_encoding.feature_extractor import EncodingInformation
from fastgres.workloads.workload import Workload
from fastgres.baseline.database_connection import DatabaseConnection
from fastgres.async_components.node import Node


class WorkloadNode(Node):

    def __init__(self, name, channel):
        super().__init__(name, channel)

    def __str__(self):
        return f"Workload Provider Node {self.name}"


def acknowledge(ch, method, properties, body):
    string_body = body.decode('utf-8')
    if string_body == 'util/ping/answer':
        print("Acknowledged Dispatcher answer")
    else:
        print(f"Unknwon body: {string_body}")
    return


def run():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "fastgres/config.ini"))
    path_config = PathConfig(path)
    workload = Workload("../fastgres/workloads/queries/stack/", "stack")
    workload.load_queries()
    database_connection = DatabaseConnection(path_config.PG_STACK_OVERFLOW, "stack_connection")
    encoding_info = EncodingInformation(database_connection, "../fastgres/database_statistics/stack/", workload)
    encoding_info.load_encoding_info()

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'fastgres/async_components/query_dispatcher.py'))
    py_patch = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            '../fastgres',
                                            '..',
                                            'venv/Scripts/python'))
    dispatcher_process = subprocess.Popen([py_patch, f"{path}", "dispatcher", "-c", "query_queue", "-dm", "batch"])
    running_processes.append(dispatcher_process)

    wl_node = WorkloadNode("Query Queue", "query_queue")
    wl_node.prepare_connection()

    wl_node.channel.basic_publish(exchange='', routing_key=wl_node.queue, body="util/ping")

    wl_node.channel.basic_publish(exchange='', routing_key=wl_node.queue, body="util/ping")


if __name__ == "__main__":
    running_processes = list()
    try:
        run()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            [_.terminate() for _ in running_processes]
            sys.exit(0)
        except SystemExit:
            [_.terminate() for _ in running_processes]
            exit(0)

