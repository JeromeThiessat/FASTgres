import abc
import pika
import threading

from typing import Any


class Node(abc.ABC):

    def __init__(self, name: str):
        self.name = name
        self.connection = None
        self.channel = None

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError


class SenderNode(Node, abc.ABC):

    def __init__(self, name: str, queues: list[str], always_open: bool = True):
        super().__init__(name)
        self.queues = queues
        self.always_open = always_open
        self.msgThread = None

        if always_open:
            self.prepare_connection()
        else:
            # Connection opening is declared manually
            pass

    def send_message(self, message: Any, queue: str):
        if queue not in self.queues:
            raise ValueError(f"Sending queue {queue} not in SenderNode declared queues")
        self.channel.basic_publish(exchange='',
                                   routing_key=queue,
                                   body=message,
                                   properties=pika.BasicProperties(delivery_mode=2))

    def prepare_connection(self):
        # self.connection = pika.SelectConnection(pika.ConnectionParameters(host="localhost"),
        #                                         on_open_callback=self._on_open)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))
        self.channel = self.connection.channel()
        self._declare_queues()

    def _declare_queues(self):
        for queue in self.queues:
            self._declare_queue(queue)

    def _declare_queue(self, queue: str):
        self.channel.queue_declare(queue=queue, durable=True)

    def close(self):
        self.connection.close()


class ReceiverNode(Node, abc.ABC):

    def __init__(self, name: str, queues: list[str]):
        super().__init__(name)
        self.queues = queues

    def prepare_connection(self):
        return pika.SelectConnection(pika.ConnectionParameters(host="localhost"),
                                     on_open_callback=self._on_open,
                                     on_close_callback=self._on_close)

    def _on_open(self, connection):
        self.channel = connection.channel(on_open_callback=self._on_channel_open)

    def _on_close(self, connection, exception):
        self.connection.ioloop.stop()

    def _on_channel_open(self, channel):
        for queue in self.queues:
            channel.basic_consume(queue, self._queue_callback, auto_ack=True)

    @abc.abstractmethod
    def _queue_callback(self, ch, method, properties, body):
        raise NotImplementedError

    def run(self):
        try:
            self.connection = self.prepare_connection()
            self.connection.ioloop.start()
        except KeyboardInterrupt:
            self.connection.close()
            self.connection.ioloop.start()
