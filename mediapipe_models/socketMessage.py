import socket

# Define connection parameters
HOST = "192.168.125.1"  # locally: 127.0.0.1  
PORT = 5000         # Same port as in RAPID

def send_socket_message(message):
    try:
        # Create a socket and connect to RobotStudio
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))

        # Send test command
        # message = "WAVE\n"
        s.sendall(message.encode())

        print("Message sent to RobotStudio:", message)

        # Close connection
        s.close()
    except Exception as e:
        print("Error:", e)
