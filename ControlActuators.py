import serial
import time

commands = [
        '0-6-i/1000-6-h/', #'0-1-i/0-6-i/0-13-i/0-9-i/0-11-i/3250-1-o/3250-6-o/3250-13-o/3250-9-o/3250-11-o/',
        'S'
    ]
index = 0

try:
    bt = serial.Serial(port='COM11', baudrate=9600, write_timeout=5)
    time.sleep(2)
    print("Connected.")

    while True:
        user_input = input(f"Enter command or press Enter to send predefined command ('{commands[index]}'): ").strip()
        if len(user_input) > 0:
            command = user_input
        else:
            command = commands[index]
            index = 1
        try:
            bt.write(command.encode())
            print(f"Command sent: {command}")
        except Exception as e:
            print(e)
        
        time.sleep(1)
        

except KeyboardInterrupt:
    print("\nCtrl-C pressed. Closing connection...")

except serial.SerialException as e:
    print(f"Could not open: {e}")

#finally:
#    if 'bt' in locals() and bt.is_open:
#        bt.close()
#        print("Connection closed.")
