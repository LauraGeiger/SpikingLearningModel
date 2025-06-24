import serial
import time

bluetooth_port = 'COM11' # or  'COM12'
#'COM10' for USB connection
baud_rate = 9600 
#115200 for USB connection

commands = [
        '0-1-i/0-6-i/0-13-i/0-9-i/0-11-i/3250-1-o/3250-6-o/3250-13-o/3250-9-o/3250-11-o/',
        'S'
    ]
index = 0

try:
    bt = serial.Serial(bluetooth_port, baud_rate)
    time.sleep(2)
    print("Connected.")

    while True:
        user_input = input(f"Enter command or press Enter to send predefined command ('{commands[index]}'): ").strip()
        if len(user_input) > 0:
            command = user_input
        else:
            command = commands[index]
            index = 1
        bt.write(command.encode())
        print(f"Command sent: {command}")
        
        time.sleep(1)

except KeyboardInterrupt:
    print("\nCtrl-C pressed. Closing connection...")

except serial.SerialException as e:
    print(f"Could not open port {bluetooth_port}: {e}")

finally:
    if 'bt' in locals() and bt.is_open:
        bt.close()
        print("Connection closed.")
