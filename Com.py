import serial
import time

# Replace with your actual COM port from Device Manager
bluetooth_port = 'COM5'  # Example: COM5, COM6, etc.
baud_rate = 115200         # Should match your Arduino's Serial.begin()

try:
    # Open serial connection
    bt = serial.Serial(bluetooth_port, baud_rate)
    time.sleep(2)  # Wait for connection to stabilize

    print("Connected. Sending command...")

    # Send command
    #command = 'A'  # You can replace this with any command string
    #bt.write(command.encode())

    # Read response (optional)
    if bt.in_waiting:
        response = bt.readline().decode().strip()
        print("Received:", response)

    bt.close()

except serial.SerialException as e:
    print(f"Could not open port {bluetooth_port}: {e}")
