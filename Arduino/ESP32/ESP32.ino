#include <AccelStepper.h>

//--- select ESP32 Dev Module ---//

// Define Pins of MUX1 --> Reading in sensors
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
    stepper.setMaxSpeed(150);       // steps/sec
    stepper.setAcceleration(50);    // steps/sec^2
}

void loop() {  
    // Read sensor values 
    for (int s = 0; s < mux1numSensors; s++) {
        for (int i = 0; i < 4; i++) {
            digitalWrite(mux1Channels[i], bitRead(s, i));
        }
        digitalWrite(mux1outputPin, LOW);
        delay(1);
        sensors[s] = analogRead(mux1potPin);
        digitalWrite(mux1outputPin, HIGH);
        delay(1);
        Serial.print(sensors[s], 2); // round to 2 decimals
        if (s < numSensors-1) Serial.print(",");  // Add comma except for last value
    }
    Serial.println();  // Newline to mark end of data packet
    delay(100);

    // Move stepper motor
    if (Serial.available()) {
        String input = Serial.readStringUntil('\n');
        input.trim(); // remove whitespace
        if (input.length() > 0) {
            int move_mm = input.toInt(); 
            long targetSteps = move_mm * STEPS_PER_MM; 
            stepper.move(targetSteps);
            Serial.printf("Moving: %d mm (%ld steps)\n", move_mm, targetSteps);
            while (stepper.distanceToGo() != 0) { 
                stepper.run(); 
            }
        }
    }
}
