"""Websocket lmb application."""
import argparse
import asyncio
import json
import logging
import signal
import sys
import websockets

logging.basicConfig(level=logging.DEBUG)
_LOGGER = logging.getLogger(__name__)

_LOGGER.warning("ASDASDASD")


class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Application:
    """Main application."""

    def __init__(self, host="ws://localhost:8080", loop=None):
        """Init."""
        _LOGGER.debug("Application init.")
        if loop is None:
            loop = asyncio.get_event_loop()

        self._host = host
        self._loop = loop
        self._socket = None
        self._tasks = Dotdict()
        self._inbox = asyncio.Queue()
        self._outbox = asyncio.Queue()

        self._tasks.main = None
        self._tasks.reader = None
        self._tasks.writer = None
        self._tasks.backbone = None

        def killer(_):
            """App killer."""
            asyncio.ensure_future(self.kill(), loop=loop)

        self._loop.add_signal_handler(signal.SIGINT, killer, None)
        self._loop.add_signal_handler(signal.SIGTERM, killer, None)
        _LOGGER.debug("Application init finished.")

    async def run(self):
        """Run application."""
        _LOGGER.debug("Starting application.")
        self._tasks.main = asyncio.Task.current_task(loop=self._loop)
        self._tasks.backbone = asyncio.ensure_future(self._backbone(), loop=self._loop)  # noqa
        while True:
            try:
                _LOGGER.debug("Connecting to: %s.", self._host)
                async with websockets.connect(self._host) as self._socket:
                    _LOGGER.debug("Established connection.")
                    try:
                        self._tasks.reader = asyncio.ensure_future(self._reader(), loop=self._loop)  # noqa  # pylint: disable=line-too-long
                        self._tasks.writer = asyncio.ensure_future(self._writer(), loop=self._loop)  # noqa  # pylint: disable=line-too-long
                        rwtasks = (self._tasks.reader, self._tasks.writer)
                        await asyncio.wait(rwtasks, return_when=asyncio.FIRST_COMPLETED)  # noqa
                    finally:
                        for task in rwtasks:
                            if not task.cancelled() and not task.done():
                                task.cancel()
                        _LOGGER.debug("Waiting for rwtasks.")
                        asyncio.wait(rwtasks)
                        _LOGGER.debug("rwtasks finished.")
            except asyncio.CancelledError:
                _LOGGER.debug("Cancelling main.")
                break
            except Exception as e:  # pylint: disable=broad-except,invalid-name
                _LOGGER.warning("Websocket exception: %s",
                                getattr(e, 'message', repr(e)))
                await asyncio.sleep(5)
            except:  # noqa  # pylint: disable=bare-except
                pass
        self._tasks.backbone.cancel()
        _LOGGER.debug("Waiting for backbone.")
        await self._tasks.backbone
        _LOGGER.debug("Leaving main.")

    async def _reader(self):
        """Reader task."""
        _LOGGER.debug("Started reader.")
        while True:
            try:
                msg = await self._socket.recv()
                _LOGGER.debug("Got message: %s.", msg)
                await self._inbox.put(msg)
            except asyncio.CancelledError:
                _LOGGER.debug("Cancelling reader task.")
                break
            except Exception as e:  # pylint: disable=broad-except,invalid-name
                _LOGGER.warning("Websocket reader exception: %s",
                                getattr(e, 'message', repr(e)))
        _LOGGER.debug("Leaving reader.")

    async def _writer(self):
        """Writer task."""
        _LOGGER.debug("Started writer.")
        while True:
            try:
                while True:
                    msg = await self._outbox.get()
                    _LOGGER.debug("Sending message: %s.", msg)
                    await self._socket.send(msg)
            except asyncio.CancelledError:
                _LOGGER.debug("Cancelling writer task.")
                break
            except Exception as e:  # pylint: disable=broad-except,invalid-name
                _LOGGER.warning("Websocket writer exception: %s",
                                getattr(e, 'message', repr(e)))
        _LOGGER.debug("Leaving writer.")

    async def _backbone(self):
        """Backbone task."""
        _LOGGER.debug("Started backbone.")
        await self.send({'a': 2, 'b': 'c'})
        while True:
            try:
                message = await self._inbox.get()
                try:
                    data = json.loads(message)
                    _LOGGER.debug("Got json: %s", repr(data))
                    await self.send(data)
                except json.JSONDecodeError as err:
                    _LOGGER.warning("Failed to parse json message: %s", err)
            except asyncio.CancelledError:
                _LOGGER.debug("Cancelling backbone task.")
                break
            except Exception as e:  # pylint: disable=broad-except,invalid-name
                _LOGGER.warning("Websocket backbone exception: %s",
                                getattr(e, 'message', repr(e)))
            except:  # noqa  # pylint: disable=bare-except
                pass
        _LOGGER.debug("Leaving backbone.")

    async def send(self, data):
        """Send JSON data."""
        await self._outbox.put(json.dumps(data))

    async def kill(self):
        """Kill application."""
        _LOGGER.debug("Killing application.")
        if self._tasks.main.cancelled():
            return
        self._tasks.main.cancel()
        await self._tasks.main
        _LOGGER.debug("Killed application.")


def parse_args(*argv):
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default="ws://localhost:8080")
    return parser.parse_args(argv)


def main(*argv):
    """Main."""
    _LOGGER.debug("Entering main.")
    args = parse_args(*argv)
    loop = asyncio.get_event_loop()
    app = Application(host=args.host, loop=loop)
    loop.run_until_complete(app.run())


if __name__ == '__main__':
    main(*sys.argv[1:])
