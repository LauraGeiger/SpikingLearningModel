#include <AccelStepper.h>

//--- select ESP32 Dev Module ---//

// Define Pins of MUX1 --> Reading in sensors
// Pressure sensor 005PGAA5 4A07-04N
// 12-bit ADC (0-4095)
const int mux1potPin = 35;  // Analog Input "SIG"
const int mux1outputPin = 19;  // Digital Output "EN"
const int mux1Channels[] = {21, 22, 23, 32};  // MUX 1 control pins

// Define Pins of MUX2 
//const int mux2potPin = 34;  // Analog Input
//const int mux2outputPin = 15;  // Digital Output
//const int mux2Channels[] = {2, 4, 5, 18};  // MUX 2 control pins

// Define Pins to control stepper motor
#define STEP_PIN   2
#define DIR_PIN    4
#define ENABLE_PIN 5  // Active LOW

// Sensor parameters
const int numSensors = 11;
const int mux1numSensors = (numSensors < 16) ? numSensors : 16;
//const int maxSensorValue = 4095;
float sensors[numSensors];

// Stepper motor parameters
const int MOTOR_STEPS = 200;                            // full steps per rev
const int MICROSTEPS = 1;                               // set by MS1-MS3 wiring on A4988
const float LEAD_MM = 5.0;                              // ~5 mm per rev
const int STEPS_PER_REV = MOTOR_STEPS * MICROSTEPS;     // 1 * 200 = 200
const float STEPS_PER_MM = STEPS_PER_REV / LEAD_MM;     // 200 / 5 = 40
static bool moving = false;                             // flag to track if motor is moving
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

void setup() {
    Serial.begin(115200);
    
    // Sensor setup
    pinMode(mux1outputPin, OUTPUT);
    digitalWrite(mux1outputPin, HIGH);
    for (int i = 0; i < 4; i++) {
        pinMode(mux1Channels[i], OUTPUT);
    }

    // Stepper motor setup
    pinMode(ENABLE_PIN, OUTPUT);
    digitalWrite(ENABLE_PIN, LOW);  // Enable driver
    stepper.setMaxSpeed(500);       // steps/sec
    stepper.setAcceleration(500);   // steps/sec^2
}

unsigned long lastSensorSend = 0;    // timestamp of last sensor packet
const unsigned long sensorInterval = 50; // ms → 20 Hz data rate

void loop() {
    unsigned long now = millis();

    // --- Stepper motor control (always runs as fast as possible) ---
    if (moving) {
        stepper.run();
        if (stepper.distanceToGo() == 0) {
            Serial.println("M:done");
            moving = false;
        }
    }

    // --- Handle incoming serial commands ---
    if (Serial.available()) {
        String input = Serial.readStringUntil('\n');
        input.trim();
        if (input.startsWith("M:")) {
            int move_mm = input.substring(2).toInt();
            long targetSteps = move_mm * STEPS_PER_MM;
            stepper.move(targetSteps);
            moving = true;
        }
    }

    // --- Sensor data sending (only every sensorInterval ms) ---
    if (now - lastSensorSend >= sensorInterval) {
        lastSensorSend = now;

        char sensor_values[128];
        int idx = 0;

        for (int s = 0; s < mux1numSensors; s++) {
            for (int i = 0; i < 4; i++) {
                digitalWrite(mux1Channels[i], bitRead(s, i));
            }
            digitalWrite(mux1outputPin, LOW);
            int val = analogRead(mux1potPin);
            digitalWrite(mux1outputPin, HIGH);

            idx += snprintf(sensor_values + idx, sizeof(sensor_values) - idx, "%d", val);
            if (s < numSensors - 1) {
                idx += snprintf(sensor_values + idx, sizeof(sensor_values) - idx, ",");
            }
        }
        Serial.println(sensor_values);
    }
}
