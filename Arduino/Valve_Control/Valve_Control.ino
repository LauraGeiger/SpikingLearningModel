#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <EEPROM.h>
#include <HoneywellTruStabilitySPI.h>


// Select ESP32 Dev Module


// Test command
// 0-2-i/0-7-i/0-9-i/0-11-i/0-13-i/0-4-i/0-1-i/0-6-i/0-8-i/500-2-h/500-7-h/500-9-h/500-11-h/500-13-h/500-4-h/500-1-h/500-6-h/500-8-h/1000-4-i/1000-1-i/1000-6-i/3000-4-h/3000-1-h/3000-6-h/


// ====================================
// Servo & Valve Configuration
// ====================================
Adafruit_PWMServoDriver pwm1 = Adafruit_PWMServoDriver(0x40);

#define NbrValves 12
#define SERVO_FREQ 50

int serINLET[NbrValves]  = {230, 192, 228, 200, 202, 215, 220, 208, 230, 170, 200, 202};
int serOUTLET[NbrValves];
int serHOLD[NbrValves];
int ServoInInlet[NbrValves] = {0};

int PumpPWM6 = 6;

// ====================================
// Servo action structure
// ====================================
struct ActionServo {
  unsigned long t;
  int servoNbr;
  char act;
  bool done;
};
ActionServo actionServo[100];
int nbr_actions = 0;
int actionIndex = 0;
bool executingActions = false;
bool sequenceOn = false;
unsigned long startTime = 0;

// ====================================
// Sensor setup
// ====================================
const unsigned int NbrS = 20;
float sval[NbrS];
TruStabilityPressureSensor sensor1(22, 0.0, 30.0);
TruStabilityPressureSensor sensor2(23, 0.0, 30.0);
TruStabilityPressureSensor sensor3(24, 0.0, 30.0);
TruStabilityPressureSensor sensor4(25, 0.0, 30.0);
TruStabilityPressureSensor sensor5(26, 0.0, 30.0);
TruStabilityPressureSensor sensor6(27, 0.0, 30.0);
TruStabilityPressureSensor sensor7(28, 0.0, 30.0);
TruStabilityPressureSensor sensor8(29, 0.0, 30.0);
TruStabilityPressureSensor sensor9(30, 0.0, 30.0);
TruStabilityPressureSensor sensor10(31, 0.0, 30.0);
TruStabilityPressureSensor sensor11(32, 0.0, 30.0);
TruStabilityPressureSensor sensor12(33, 0.0, 30.0);
TruStabilityPressureSensor sensor13(34, 0.0, 30.0);
TruStabilityPressureSensor sensor14(35, 0.0, 30.0);
TruStabilityPressureSensor sensor15(36, 0.0, 30.0);
TruStabilityPressureSensor sensor16(37, 0.0, 30.0);
TruStabilityPressureSensor sensor17(38, 0.0, 30.0);
TruStabilityPressureSensor sensor18(39, 0.0, 30.0);
TruStabilityPressureSensor sensor19(40, 0.0, 30.0);
TruStabilityPressureSensor sensor20(41, 0.0, 30.0);

// ====================================
// EEPROM functions
// ====================================
void StorInletInEEPROM() {
  Serial.println("EEPROM storing start..");
  for (int i = 0; i < NbrValves; i++) {
    EEPROM.write(i*2 , highByte(serINLET[i]));
    EEPROM.write(i*2+1, lowByte(serINLET[i]));
  }
  Serial.println("EEPROM end storing ..");
  delay(2000); 
}

void RetriveInletFromEEPROM() {
  for (int i = 0; i < NbrValves; i++) {
    serINLET[i] = word(EEPROM.read(i*2), EEPROM.read(i*2+1));
    Serial.print("Reading one word(");
    Serial.print(serINLET[i]);
    Serial.print(") from EEPROM address: ");
    Serial.println(i*2);
  }
}

// ====================================
// Sensor reading
// ====================================
void readSensors() {
  if(sensor1.readSensor() == 0) sval[0] = sensor1.pressure();
  if(sensor2.readSensor() == 0) sval[1] = sensor2.pressure();
  if(sensor3.readSensor() == 0) sval[2] = sensor3.pressure();
  if(sensor4.readSensor() == 0) sval[3] = sensor4.pressure();
  if(sensor5.readSensor() == 0) sval[4] = sensor5.pressure();
  if(sensor6.readSensor() == 0) sval[5] = sensor6.pressure();
  if(sensor7.readSensor() == 0) sval[6] = sensor7.pressure();
  if(sensor8.readSensor() == 0) sval[7] = sensor8.pressure();
  if(sensor9.readSensor() == 0) sval[8] = sensor9.pressure();
  if(sensor10.readSensor() == 0) sval[9] = sensor10.pressure();
  if(sensor11.readSensor() == 0) sval[10] = sensor11.pressure();
  if(sensor12.readSensor() == 0) sval[11] = sensor12.pressure();
  if(sensor13.readSensor() == 0) sval[12] = sensor13.pressure();
  if(sensor14.readSensor() == 0) sval[13] = sensor14.pressure();
  if(sensor15.readSensor() == 0) sval[14] = sensor15.pressure();
  if(sensor16.readSensor() == 0) sval[15] = sensor16.pressure();
  if(sensor17.readSensor() == 0) sval[16] = sensor17.pressure();
  if(sensor18.readSensor() == 0) sval[17] = sensor18.pressure();
  if(sensor19.readSensor() == 0) sval[18] = sensor19.pressure();
  if(sensor20.readSensor() == 0) sval[19] = sensor20.pressure();
}

void printSensors() {
  for(int i=0;i<NbrS;i++){
    Serial.print(sval[i]);
    Serial.print("\t");
  }
  Serial.println();
  delay(100);
}

// ====================================
// Servo functions
// ====================================
void setupServos() {
  pwm1.begin();
  pwm1.setOscillatorFrequency(27000000);
  pwm1.setPWMFreq(SERVO_FREQ);

  for (int i = 0; i < NbrValves; i++) {
    serOUTLET[i] = serINLET[i] + 180;
    serHOLD[i] = (serINLET[i] + serOUTLET[i]) / 2;
    pwm1.setPWM(i, 0, serOUTLET[i]);
    Serial.print("set servo: ");
    Serial.println(i+1);
    delay(200);
  }
  pwm1.sleep();
}

void startSequence() {
  if (actionIndex > 0) {
    nbr_actions = actionIndex;
    actionIndex = 0;
  }
  for (int i = 0; i < nbr_actions; i++) {
    actionServo[i].done = false;
    
    Serial.print(actionServo[i].t);
    Serial.print("\t");
    Serial.print(actionServo[i].servoNbr);
    Serial.print("\t");
    Serial.print(actionServo[i].act);
    Serial.print("\n");
  }
  Serial.print("nbr_actions: "); 
  Serial.println(nbr_actions);
  startTime = millis();
  executingActions = true;
  Serial.println("Sequence started!");
}

void stopSequence() {
  executingActions = false;
  pwm1.wakeup();
  for (int i = 0; i < NbrValves; i++) {
      pwm1.setPWM(i, 0, serOUTLET[i]);
      ServoInInlet[i] = 0;
  }
  delay(200);
  pwm1.sleep();
  analogWrite(PumpPWM6, 0);
  Serial.println("Sequence stopped, all valves to OUTLET");
}

void toggleSequence() {
  sequenceOn = !sequenceOn;
  if (sequenceOn) startSequence();
  else stopSequence();
}

void updateServos() {
  if (!executingActions) return;
  unsigned long now = millis() - startTime;
  int InletSum = 0;
  pwm1.wakeup();
  for (int i = 0; i < nbr_actions; i++) {
    if (!actionServo[i].done && now >= actionServo[i].t) {
      int val;
      switch (actionServo[i].act) {
        case 'i': val = serINLET[actionServo[i].servoNbr]; ServoInInlet[actionServo[i].servoNbr] = 1; break;
        case 'o': val = serOUTLET[actionServo[i].servoNbr]; ServoInInlet[actionServo[i].servoNbr] = 0; break;
        case 'h': val = serHOLD[actionServo[i].servoNbr]; ServoInInlet[actionServo[i].servoNbr] = 0; break;
        default: val = serOUTLET[actionServo[i].servoNbr]; ServoInInlet[actionServo[i].servoNbr] = 0; break;
      }
      pwm1.setPWM(actionServo[i].servoNbr, 0, val);
      actionServo[i].done = true;
    }
    InletSum += ServoInInlet[actionServo[i].servoNbr];
  }
  delay(200);
  analogWrite(PumpPWM6, InletSum > 0 ? 255 : 0);

  bool allDone = true;
  for (int i = 0; i < nbr_actions; i++) if (!actionServo[i].done) allDone = false;
  if (allDone) {
    pwm1.sleep();
    executingActions = false;
  }
}

// ====================================
// Serial reading
// ====================================
#define TOKEN_BUFFER_SIZE 64
char tokenBuffer[TOKEN_BUFFER_SIZE];
int tokenIndex = 0;

void parseToken(const char* token) {
  // Handle special commands
  Serial.println(token);
  if (strcmp(token, "S") == 0) {
    toggleSequence();
    return;
  } else if (strcmp(token, "Start") == 0) {
    startSequence();
    return;
  } else if (strcmp(token, "Stop") == 0) {
    stopSequence();
    return;
  }

  // Parse servo action: time-servo-act
  int dash1 = -1, dash2 = -1;
  for (int i = 0; token[i] != '\0'; i++) {
    if (token[i] == '-' && dash1 == -1) dash1 = i;
    else if (token[i] == '-' && dash2 == -1) dash2 = i;
  }
  if (dash1 == -1 || dash2 == -1) return;

  String timeStr   = String(token).substring(0, dash1);
  String servoStr  = String(token).substring(dash1 + 1, dash2);
  String actionStr = String(token).substring(dash2 + 1);

  if (actionIndex < 100) {
    actionServo[actionIndex].t = timeStr.toInt();
    actionServo[actionIndex].servoNbr = servoStr.toInt() - 1;
    actionServo[actionIndex].act = actionStr[0];
    actionServo[actionIndex].done = false;
    actionIndex++;
  }
}

void processChar(char c) {
  Serial.println(c);
  if (c == '\n' || c == '/') {
    if (tokenIndex > 0) {
      tokenBuffer[tokenIndex] = '\0';   // terminate string
      parseToken(tokenBuffer);          // parse immediately
      tokenIndex = 0;
    }
  } else if (c != '\r') {  // ignore CR
    if (tokenIndex < TOKEN_BUFFER_SIZE - 1) {
      tokenBuffer[tokenIndex++] = c;
    } else {
      Serial.println("ERROR: token too long!");
      tokenIndex = 0; // reset on overflow
    }
  }
}

void readSerial() {
  while (Serial.available()) {
    processChar(Serial.read());
  }
  while (Serial3.available()) {
    processChar(Serial3.read());
  }
}

// ====================================
// Arduino setup & loop
// ====================================
void setup() {
  Serial.begin(115200);
  Serial3.begin(9600);

  pinMode(PumpPWM6, OUTPUT);
  setupServos();
  analogWrite(PumpPWM6, 0);

  SPI.begin();
  sensor1.begin(); sensor2.begin(); sensor3.begin(); sensor4.begin(); sensor5.begin();
  sensor6.begin(); sensor7.begin(); sensor8.begin(); sensor9.begin(); sensor10.begin();
  sensor11.begin(); sensor12.begin(); sensor13.begin(); sensor14.begin(); sensor15.begin();
  sensor16.begin(); sensor17.begin(); sensor18.begin(); sensor19.begin(); sensor20.begin();
}

void loop() {
  readSerial();   // continuously read USB and Bluetooth
  updateServos(); // non-blocking servo execution
  readSensors();  // read sensors without blocking
  //printSensors(); // print sensor values
  delay(1);       // small delay to avoid flooding Serial
}
