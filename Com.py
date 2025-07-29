import serial
import time

commands = [
        '0-6-i/1000-6-h/', #'0-1-i/0-6-i/0-13-i/0-9-i/0-11-i/3250-1-o/3250-6-o/3250-13-o/3250-9-o/3250-11-o/',
        'S'
    ]
index = 0

def read_sensor_data():
    if ser.in_waiting > 0:
        while ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
        values = [float(x) for x in line.split(',')]
        print(f"sensor values = {values}")
        #diff = [round(v - b, 2) for v, b in zip(values, sensor_baseline)]
        #print(f"sensor diifs = {diff}")

try:
    bt = serial.Serial(port='COM11', baudrate=9600, write_timeout=5)
    ser = serial.Serial(port='COM7', baudrate=115200, timeout=1) # For ESP32
    time.sleep(2)
    print("Connected.")

    for dur in range(0,5):
        print(f"Actuate index finger flexion for {dur}s")
        bt.write(f'0-6-i/{dur*1000}-6-h/'.encode())
        time.sleep(3)
        print("Start")
        bt.write(f'S'.encode())
        time.sleep(1)
        read_sensor_data()
        time.sleep(3+dur)
        print("Stop")
        bt.write(f'S'.encode())
        time.sleep(5)

    '''
    movement = False

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
            if command == 'S' and movement:
                read_sensor_data()

            movement = not movement
        except Exception as e:
            print(e)
        
        time.sleep(1)
        '''

except KeyboardInterrupt:
    print("\nCtrl-C pressed. Closing connection...")

except serial.SerialException as e:
    print(f"Could not open: {e}")

finally:
    if 'bt' in locals() and bt.is_open:
        bt.close()
        print("Connection closed.")
