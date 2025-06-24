import serial
import time
import threading
import re

bluetooth_port = 'COM12' #'COM10' for USB connection
baud_rate = 9600 #115200 for USB connection

set_servos = set()
lock = threading.Lock()  # To safely update set_servos from both threads

def read_from_serial(bt):
    global set_servos
    try:
        while True:
            if bt.in_waiting > 0:
                data = bt.read(bt.in_waiting)
                text = data.decode('latin-1', errors='ignore')
                print(text, end='')

                # Detect "set servo: X" messages
                matches = re.findall(r'set servo:\s*(\d+)', text)
                with lock:
                    for m in matches:
                        set_servos.add(int(m))

            time.sleep(0.1)
    except serial.SerialException:
        print("Serial port error or disconnected.")
    except Exception as e:
        print(f"Error in reading thread: {e}")

try:
    bt = serial.Serial(bluetooth_port, baud_rate)
    time.sleep(2)

    print("Connected. Receiving messages...")

    #reader_thread = threading.Thread(target=read_from_serial, args=(bt,), daemon=True)
    #reader_thread.start()

    commands = [
        '0-1-i/0-6-i/0-13-i/0-9-i/0-11-i/3250-1-o/3250-6-o/3250-13-o/3250-9-o/3250-11-o/',
        'S'
    ]
    index = 0

    while True:
        input("Press Enter to send command...")

        with lock:
        #    if set_servos >= set(range(1, 13)):
            command = commands[index]
            bt.write(command.encode())
            print(f"\nCommand sent: {command}")
            index = 1
            time.sleep(1)
        #    else:
        #        print("Not all servos are set yet. Command not sent.")

except KeyboardInterrupt:
    print("\nCtrl-C pressed. Closing connection...")

except serial.SerialException as e:
    print(f"Could not open port {bluetooth_port}: {e}")

finally:
    if 'bt' in locals() and bt.is_open:
        bt.close()
        print("Connection closed.")
