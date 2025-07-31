
import serial
import time

# Valve - Actuators
#  1 - Thumb flexor
#  2 - Thumb extensor
#  3 - Thumb abductor
#  4 - Thumb oppositor
#  5 - Wrist dorsifl
#  6 - Index finger flexor
#  7 - Index finger extensor
#  8 - Middle finger flexor
#  9 - Middle finger extensor
# 10 - Ring finger flexor
# 11 - Ring finger extensor
# 12 - Pinky finger flexor
# 13 - Pinky finger extensor

time1 = 1000
time2 = 1400
time3 = 3000
time4 = 5000
time5 = 7000
commands = [
         f'{time1}-1-i/{time1}-2-i/{time1}-3-i/{time1}-4-i/{time1}-7-i/{time1}-6-i/{time1}-13-i/{time1}-12-i/{time1}-9-i/{time1}-8-i/{time1}-11-i/{time1}-10-i/'
        +f'{time2}-1-h/{time2}-2-h/{time2}-3-h/{time2}-4-h/{time2}-7-h/{time2}-6-h/{time2}-13-h/{time2}-12-h/{time2}-9-h/{time2}-8-h/{time2}-11-h/{time2}-10-h/'
        +f'{time3}-1-i/{time3}-4-i/{time3}-6-i/'
        +f'{time4}-4-h/'
        +f'{time5}-1-h/{time5}-6-h/',
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
