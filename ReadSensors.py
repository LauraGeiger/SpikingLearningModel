import serial
import time

def read_sensor_data_flex(duration=5):
    ser_sensor.flushInput() # delete values in serial input buffer
    recorded_data = []  # Stores all sensor readings
    print("Reading sensor data...")
    start_time = time.time()
    while time.time() - start_time < duration:
        if ser_sensor.in_waiting > 0:
            line = ser_sensor.readline().decode('utf-8', errors='ignore').strip()
            try:
                values = [float(x) for x in line.split(',')]
                recorded_data.append(values)
            except ValueError:
                print(f"Ignored malformed line: {line}")
        time.sleep(0.01)  # ~100 Hz sampling
    print("Sensor data collection complete.")
    return recorded_data

def analyze_sensor_data_flex(data, alpha=0.8, flexion_threshold=30, extension_threshold=30):
    num_sensors = len(data[0])
    prev_filtered = [data[0][i] for i in range(num_sensors)]
    start_filtered = prev_filtered.copy()
    max_filtered = prev_filtered.copy()
    min_filtered = prev_filtered.copy()

    flexion_detected = [False] * num_sensors
    extension_detected = [False] * num_sensors

    for sample in data[1:]:
        for i in range(num_sensors):
            # Apply low-pass filter
            filtered = alpha * prev_filtered[i] + (1 - alpha) * sample[i]

            # Track max and min
            max_filtered[i] = max(max_filtered[i], filtered)
            min_filtered[i] = min(min_filtered[i], filtered)

            # Detect flexion and extension
            if (max_filtered[i] - start_filtered[i]) > flexion_threshold:
                flexion_detected[i] = True
            if (start_filtered[i] - min_filtered[i]) > extension_threshold:
                extension_detected[i] = True

            prev_filtered[i] = filtered

    # Print results
    print("\nFlexion and Extension Detection Results:")
    for i in range(num_sensors):
        baseline = start_filtered[i]
        flex = flexion_detected[i]
        extend = extension_detected[i]
        delta_up = max_filtered[i] - start_filtered[i]
        delta_down = start_filtered[i] - min_filtered[i]
        print(
            f"Sensor {i}: "
            f"Baseline = {baseline} "
            f"{'👉 Flexion' if flex else '   '} "
            f"{'👈 Extension' if extend else ''} "
            f"(Δ up = {delta_up:.2f}, Δ down = {delta_down:.2f})"
        )


try:
    bt = serial.Serial(port='COM11', baudrate=9600, write_timeout=5)
    ser_sensor = serial.Serial(port='COM7', baudrate=115200, timeout=1) # For ESP32
    time.sleep(2)
    print("Connected.")

    
    dur = 4
    print(f"Actuate index finger flexion for {dur}s")
    bt.write(f'0-6-i/{dur*1000}-6-h/'.encode())
    time.sleep(2)

    bt.write(f'S'.encode())  # Start device
    recorded_data_1 = read_sensor_data_flex(duration=2+dur)

    bt.write(f'S'.encode())  # Stop device
    recorded_data_2 = read_sensor_data_flex(duration=2+dur)

    


    print("Analyzing data 1...")
    analyze_sensor_data_flex(recorded_data_1)
    print("Analyzing data2...")
    analyze_sensor_data_flex(recorded_data_2)



except KeyboardInterrupt:
    print("\nCtrl-C pressed. Closing connection...")

except serial.SerialException as e:
    print(f"Could not open: {e}")

#finally:
#    if 'bt' in locals() and bt.is_open:
#        bt.close()
#        print("Connection closed.")
