import socket
import threading
import time

class NotificationListener:
    def __init__(self,
                 host: str,
                 port: int,
                 on_message=None,
                 retry_delay: float = 1.0,
                 recv_timeout: float = 0.1,
                 delimiter: str = '\n'):   # match whatever RAPID is using
        self.host = host
        self.port = port
        self.on_message = on_message or (lambda msg: print("Notification:", msg))
        self.retry_delay = retry_delay
        self.recv_timeout = recv_timeout
        self.delimiter = delimiter

        self._stop = threading.Event()
        self._sock = None

        print(f"[Notifier] Initializing listener for {host}:{port}, delim={repr(delimiter)}")
        self._connect_loop()

        print("[Notifier] Starting receive thread")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _connect_loop(self):
        while not self._stop.is_set():
            try:
                print(f"[Notifier] Attempting connect to {self.host}:{self.port} …")
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(self.recv_timeout)
                s.connect((self.host, self.port))
                self._sock = s
                print(f"NotificationListener: connected to {self.host}:{self.port}")
                return
            except Exception as e:
                print(f"NotificationListener: connect failed ({e}), retrying in {self.retry_delay}s")
                time.sleep(self.retry_delay)

    def _run(self):
        buf = ""
        sock = self._sock
        print("[Notifier] Receive loop entered")
        while not self._stop.is_set():
            try:
                data = sock.recv(1024)
                if not data:
                    print("[Notifier] Socket closed by server")
                    break
                # Show raw bytes so we can see exactly what’s arriving
                print(f"[Notifier] Raw bytes: {data!r}")
                text = data.decode("utf8", "ignore")
                buf += text

                # Now split on the delimiter you chose
                while self.delimiter in buf:
                    line, buf = buf.split(self.delimiter, 1)
                    line = line.strip()
                    if line:
                        print(f"[Notifier] Complete msg: {line!r}")
                        self.on_message(line)

            except socket.timeout:
                continue
            except Exception as e:
                print(f"[Notifier] Recv error: {e!r}")
                break

    def stop(self):
        """Signal the thread to exit and wait for it."""
        self._stop.set()
        if self._sock:
            self._sock.close()
        self.thread.join()
        print("NotificationListener: stopped")
