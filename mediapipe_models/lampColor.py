
import requests

#ESP32_IP = "http://192.168.195.233" # Test board
ESP32_IP = "http://192.168.195.9" # Arduino board at school

# List of commands that can be sent ["RED", "GREEN", "BLUE", "EXIT"]

def change_lamp_color(message):
    color = "GREEN"
    match message:
        case "Closed_Fist" | "Middle_Finger":
            color = "RED"
        case "Thumb_Down":
            color = "BLUE"
        case _:
            color = "GREEN"
    send_color_change(color)

def send_color_change(color):
    try:
        response = requests.get(f"{ESP32_IP}/{color}", timeout=0.5)
        print(f"Response from ESP32 for {color}:")
        print(response.text)
    except requests.exceptions.RequestException as e:
        print("Error sending request:", e)