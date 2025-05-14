import socket
import time
from lampColor import change_lamp_color

class RobotSocketClient:
    def __init__(self, host='192.168.125.1', port=5000, retry_delay=0):
        self.host = host
        self.port = port
        self.retry_delay = retry_delay
        self.sock = None

    def connect(self):
        while self.sock is None:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(0.1) 
                self.sock.connect((self.host, self.port))
                print("Connected to RobotStudio.")
            except Exception as e:
                print("Connection failed, retrying in", self.retry_delay, "seconds. Error:", e)
                self.sock = None
                time.sleep(self.retry_delay)

    def send_message(self, message):
        if not message or not message.strip():
            print("No message to send.")
            return

        if self.sock is None:
            self.connect()

        try:
            sent = self.sock.send(message.encode())
            change_lamp_color(message)
            if sent == 0:
                raise RuntimeError("socket connection broken")
            print("Message sent to RobotStudio:", message)
        except (BlockingIOError, socket.timeout):
            # the socket would have blocked, so just drop this frame's message
            print("Send would block or timed out; skipping this gesture.")
        except Exception as e:
            print("Send failed:", e)
            self.sock.close()
            self.sock = None
            self.connect()

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None
            print("Connection closed.")
