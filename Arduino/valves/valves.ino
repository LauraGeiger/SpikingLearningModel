#include <HoneywellTruStabilitySPI.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <EEPROM.h>



// 0-1-i/0-2-i/0-3-i/0-4-i/0-5-i/0-6-i/0-7-i/0-8-i/0-9-i/0-10-i/0-11-i/0-12-i/0-13-i/#
// 1000-1-h/1000-2-h/1000-3-h/1000-4-h/1000-5-h/1000-6-h/1000-7-h/1000-8-h/1000-9-h/1000-10-h/1000-11-h/1000-12-h/1000-13-h/#
// 0-1-i/0-2-i/0-3-i/0-4-i/0-5-i/0-6-i/0-7-i/0-8-i/0-9-i/0-10-i/0-11-i/0-12-i/0-13-i/1000-1-h/1000-2-h/1000-3-h/1000-4-h/1000-5-h/1000-6-h/1000-7-h/1000-8-h/1000-9-h/1000-10-h/1000-11-h/1000-12-h/1000-13-h/#

Adafruit_PWMServoDriver pwm1 = Adafruit_PWMServoDriver(0x40);

const unsigned int NbrS = 20;
float sval[NbrS];
TruStabilityPressureSensor sensor1( 22, 0.0, 30.0 );
TruStabilityPressureSensor sensor2( 23, 0.0, 30.0 );
TruStabilityPressureSensor sensor3( 24, 0.0, 30.0 );
TruStabilityPressureSensor sensor4( 25, 0.0, 30.0 );
TruStabilityPressureSensor sensor5( 26, 0.0, 30.0 );
TruStabilityPressureSensor sensor6( 27, 0.0, 30.0 );
TruStabilityPressureSensor sensor7( 28, 0.0, 30.0 );
TruStabilityPressureSensor sensor8( 29, 0.0, 30.0 );
TruStabilityPressureSensor sensor9( 30, 0.0, 30.0 );
TruStabilityPressureSensor sensor10(31, 0.0, 30.0 );

TruStabilityPressureSensor sensor11( 32, 0.0, 30.0 );
TruStabilityPressureSensor sensor12( 33, 0.0, 30.0 );
TruStabilityPressureSensor sensor13( 34, 0.0, 30.0 );
TruStabilityPressureSensor sensor14( 35, 0.0, 30.0 );
TruStabilityPressureSensor sensor15( 36, 0.0, 30.0 );
TruStabilityPressureSensor sensor16( 37, 0.0, 30.0 );
TruStabilityPressureSensor sensor17( 38, 0.0, 30.0 ); 
TruStabilityPressureSensor sensor18( 39, 0.0, 30.0 );
TruStabilityPressureSensor sensor19( 40, 0.0, 30.0 );
TruStabilityPressureSensor sensor20( 41, 0.0, 30.0 );

#define SERVOMIN  150 // This is the 'minimum' pulse length count (out of 4096)
#define SERVOMAX  600 // This is the 'maximum' pulse length count (out of 4096)
#define USMIN  600 // This is the rounded 'minimum' microsecond length based on the minimum pulse of 150
#define USMAX  2400 // This is the rounded 'maximum' microsecond length based on the maximum pulse of 600
#define SERVO_FREQ 50 // Analog servos run at ~50 Hz updates

#define DONOTHING_GLOVE 0 
#define ACTUATING_GLOVE 1
#define HOLDING_GLOVE   2
#define RELAXING_GLOVE  3

byte glove_state = RELAXING_GLOVE;

const int NbrValves = 12;

int v = 200;
int w = 380;

//int serINLET[NbrValves]  ={v,v,v,v,v,v,v,v,v,v,v,v,v};
//int serOUTLET[NbrValves] ={w,w,w,w,w,w,w,w,w,w,w,w,w};


//int  serINLET[NbrValves];
//int serINLET[NbrValves] ={v+30,v-10,v+30,v,v,v+15,v+20,v+10,v+30,v-30,v,v}; //Blue 1 
int serINLET[NbrValves] ={v+30,v-8,v+28,v,v+2,v+15,v+20,v+8,v+30,v-30,v,v+2}; //Blue 1 
int serOUTLET[NbrValves];
int serHOLD[NbrValves];

int ServoInInlet[NbrValves]={0}; // this numbers are needed to know how many servo are in inlet position at the current time

int pos = 0;      // position in degrees

int PumpPWM6 = 6;      // LED connected to digital pin 9
int PumpPWM7 = 7;   // potentiometer connected to analog pin 3

int dval = 1;
int SerialVal =1; 


//========================================================
void StorInletInEEPROM()
{
  Serial.print( "EEPROM storing start.. \n");
  for (int i =0;i<NbrValves;i++)
  {
    EEPROM.write(i*2 , highByte(serINLET[i] ));
    EEPROM.write(i*2+1, lowByte(serINLET[i] ));
  }
  Serial.print( "EEPROM end storing .. \n");
  delay(2000); 
}
//========================================================
void RetriveInletFromEEPROM()
{
  for (int i =0;i<NbrValves;i++)
  {
    serINLET[i]=  word(EEPROM.read(i*2), EEPROM.read(i*2+1));
    Serial.print("Reading one word(");
    Serial.print(serINLET[i]);
    Serial.print(" from EEPROM address:");
    Serial.println(i*2);
  }
  delay(5000); 
}
//========================================================
String getValue(String data, char separator, int index)
{
  int found = 0;
  int strIndex[] = {0, -1};
  int maxIndex = data.length()-1;

  for(int i=0; i<=maxIndex && found<=index; i++){
    if(data.charAt(i)==separator || i==maxIndex){
      found++;
      strIndex[0] = strIndex[1]+1;
      strIndex[1] = (i == maxIndex) ? i+1 : i;
    }
  }
  return found>index ? data.substring(strIndex[0], strIndex[1]) : "";
}
//========================================================
void readSensors()
{
  // the sensor returns 0 when new data is ready
  if( sensor1.readSensor() == 0 )    sval[0]=sensor1.pressure();
  if( sensor2.readSensor() == 0 )    sval[1]=sensor2.pressure();
  if( sensor3.readSensor() == 0 )    sval[2]=sensor3.pressure();
  if( sensor4.readSensor() == 0 )    sval[3]=sensor4.pressure();
  if( sensor5.readSensor() == 0 )    sval[4]=sensor5.pressure();
  if( sensor6.readSensor() == 0 )    sval[5]=sensor6.pressure();
  if( sensor7.readSensor() == 0 )    sval[6]=sensor7.pressure();
  if( sensor8.readSensor() == 0 )    sval[7]=sensor8.pressure();
  if( sensor9.readSensor() == 0 )    sval[8]=sensor9.pressure();
  if( sensor10.readSensor() == 0 )   sval[9]=sensor10.pressure();

  if( sensor11.readSensor() == 0 )    sval[10]=sensor11.pressure();
  if( sensor12.readSensor() == 0 )    sval[11]=sensor12.pressure();
  if( sensor13.readSensor() == 0 )    sval[12]=sensor13.pressure();
  if( sensor14.readSensor() == 0 )    sval[13]=sensor14.pressure();
  if( sensor15.readSensor() == 0 )    sval[14]=sensor15.pressure();
  if( sensor16.readSensor() == 0 )    sval[15]=sensor16.pressure();
  if( sensor17.readSensor() == 0 )    sval[16]=sensor17.pressure();
  if( sensor18.readSensor() == 0 )    sval[17]=sensor18.pressure();
  if( sensor19.readSensor() == 0 )    sval[18]=sensor19.pressure();
  if( sensor20.readSensor() == 0 )    sval[19]=sensor20.pressure();
}

void PrintSensors()
{
  for(int i=0;i<NbrS;i++)
  {
    Serial.print(sval[i]);
    Serial.print("\t");
  }
  Serial.print("\n");
}

//========================================================
void mydelay(long unsigned d)
{
  unsigned long previousMillis = millis();
  while(millis() - previousMillis < d)
  {
    readSensors();//readUDP(); 
    //PrintSensors();
    delay(1);
  }
}
//========================================================
void setup() {
  // put your setup code here, to run once:

  Serial.begin(115200); // start Serial communication
  Serial3.begin(9600);

  pinMode(PumpPWM6, OUTPUT);  // sets the pin as output
  pinMode(PumpPWM7, OUTPUT);  // sets the pin as output
  analogWrite(PumpPWM7, 0); // analogRead values go from 0 to 1023, analogWrite values from 0 to 255

  for(int i=0;i< (22+NbrS);i++)
  {
    pinMode(i+22, OUTPUT);
    digitalWrite(i+22, HIGH);
  }
  SPI.begin(); // start SPI communication
  sensor1.begin(); // run sensor initialization
  sensor2.begin(); // run sensor initialization
  sensor3.begin(); // run sensor initialization
  sensor4.begin(); // run sensor initialization
  sensor5.begin(); // run sensor initialization
  sensor6.begin(); // run sensor initialization
  sensor7.begin(); // run sensor initialization
  sensor8.begin(); // run sensor initialization
  sensor9.begin(); // run sensor initialization
  sensor10.begin(); // run sensor initialization
  sensor11.begin(); // run sensor initialization
  sensor12.begin(); // run sensor initialization
  sensor13.begin(); // run sensor initialization
  sensor14.begin(); // run sensor initialization
  sensor15.begin(); // run sensor initialization
  sensor16.begin(); // run sensor initialization
  sensor17.begin(); // run sensor initialization
  sensor18.begin(); // run sensor initialization
  sensor19.begin(); // run sensor initialization
  sensor20.begin(); // run sensor initialization
  delay(1000);

  pwm1.begin();
  pwm1.setOscillatorFrequency(27000000);
  pwm1.setPWMFreq(SERVO_FREQ);  // Analog servos run at ~50 Hz updates
  pwm1.sleep();

  //RetriveInletFromEEPROM();

  for (int i =0;i<NbrValves;i++)
  {
    serOUTLET[i]=serINLET[i]+180;
    serHOLD[i]= (serOUTLET[i]+serINLET[i])/2.0;
  }

  pwm1.wakeup();
  for (int i =0;i<NbrValves;i++)
  {
    pwm1.setPWM(i, 0, serOUTLET[i]);
    Serial.print("set servo: ");
    Serial.println(i+1);
    delay(200);
  }
  delay(1000);
  pwm1.sleep();

}
//========================================================

int i='f';

struct actionServo {
  unsigned int t;
  unsigned int servoNbr;
  char act; 
} 
actionServo[100];

int in, in2, in3, in4;
int nbr_actions = 0;
int ServoVal;


//========================================================
void loop() {
  // put your main code here, to run repeatedly:

  String str, timeStr, servoStr, actionStr;

  if((Serial3.available() > 0)||(Serial.available() > 0)) 
  {
    if(Serial3.available() > 0) {
      //str= Serial3.readString(); 
      str= Serial3.readStringUntil('#');
    }
    if(Serial.available() > 0) {
      //str= Serial.readString(); 
      str= Serial.readStringUntil('#');
    }

    if(str[0]=='E') // EEPROM store
    {
      SerialVal=0; 
      dval=0;
      StorInletInEEPROM(); 
      Serial.print( "E pressed..\n");
    }
    else if(str[0]=='T')
    {
      SerialVal=0; 
      dval=0;
      glove_state = RELAXING_GLOVE;
      Serial.print( "T pressed..\n");
    }
    else if(str[0]=='S')
    {
      SerialVal=0; 
      dval=0;
      Serial.print( "S pressed..\n");
    }
    else
    {
      SerialVal=1;  
      in=0; // start from first index 
      nbr_actions = 0;
      Serial.println(str);
      while(1)
      {
        in2 = str.indexOf("/",in);
        in3 = str.indexOf("-",in);
        in4 = str.indexOf("-",in3+1);
      
        if (in2==-1)
          break;

        timeStr = str.substring(in, in3);
        servoStr = str.substring(in3+1, in4);
        actionStr = str.substring(in4+1, in2);

        //Serial.print(timeStr); Serial.print("\t");  
        //Serial.print(servoStr); Serial.print("\t");  
        //Serial.print(actionStr); Serial.print("\n");  
        
        in=in2+1; // search from next index

        actionServo[nbr_actions].t=timeStr.toInt();
        actionServo[nbr_actions].servoNbr=servoStr.toInt()-1; // indexing from matlab to C 
        actionServo[nbr_actions].act=actionStr[0];

        nbr_actions++;
      }
      Serial.print( "nbr_actions:"); 
      Serial.print( nbr_actions); 
      Serial.print("\n");  


      for(int i=0;i<nbr_actions;i++)
      {
        Serial.print(actionServo[i].t);
        Serial.print("\t");
        Serial.print(actionServo[i].servoNbr);
        Serial.print("\t");
        Serial.print(actionServo[i].act);
        Serial.print("\n");
      }
    }
  }

  if ((dval==0)&&(glove_state == RELAXING_GLOVE))
  {
    glove_state = HOLDING_GLOVE;
    dval=1;

    pwm1.wakeup();
    mydelay(50);

    for(int i=0 ; i<nbr_actions ; i++)
    {
      if(actionServo[i].act=='i')
      {
        ServoVal= serINLET[actionServo[i].servoNbr]; 
        ServoInInlet[actionServo[i].servoNbr]=1;
      }
      if(actionServo[i].act=='o')
      {
        ServoVal= serOUTLET[actionServo[i].servoNbr]; 
        ServoInInlet[actionServo[i].servoNbr]=0;
      }
      if(actionServo[i].act=='h')
      {
        ServoVal= serHOLD[actionServo[i].servoNbr]; 
        ServoInInlet[actionServo[i].servoNbr]=0;
      }

      int InletSum=0;
      for (int j=0;j<NbrValves;j++)
        InletSum+=ServoInInlet[j];

      if(InletSum>0)
        analogWrite(PumpPWM6, 255);
      else 
        analogWrite(PumpPWM6, 0);

      Serial.print("InletSum:");
      Serial.println(InletSum);

      pwm1.setPWM(actionServo[i].servoNbr, 0, ServoVal);

      if(i<(nbr_actions-1)) {
        Serial.print("Delay:");
        Serial.println(actionServo[i+1].t - actionServo[i].t);
        delay(actionServo[i+1].t - actionServo[i].t);
      }

    }
    mydelay(10);
  }

  if ((dval==0)&&(glove_state == HOLDING_GLOVE))
  {
    glove_state = RELAXING_GLOVE;
    dval=1;
    analogWrite(PumpPWM6, 0);
    pwm1.wakeup();
    mydelay(100);    
    for (int i =0;i<NbrValves;i++)
    {
      pwm1.setPWM(i, 0, serOUTLET[i]);
      delay(10);
    }
    mydelay(500);
    //mydelay(2000);
    pwm1.sleep();
    mydelay(50);
  } 
  mydelay(10);
}
