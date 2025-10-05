// Author: Endri Dibra 
// Bachelor Thesis: Smart Unmanned Ground Vehicle

// Importing the required libraries 
#include <DHT.h>
#include <math.h>
#include <Wire.h>
#include <Servo.h>
#include <MPU6050.h>
#include <ArduinoJson.h>


// Defining DHT Sensor Pin
#define DHT_PIN 11

// Defining DHT Type
#define DHT_TYPE DHT11

// Implementing DHT11 initialization
DHT dht(DHT_PIN, DHT_TYPE);

// Defining Gas Sensor Pin
#define GAS_PIN A0

// Defining DHT11 sensor reading frequency
// every 5000 ms or 5 seconds
const long DHT11FREQUENCY = 5000;

// Defining last reading from the DHT11 sensor
unsigned long lastDHT11Read = 0;

// Defining Servo motor pin
#define SERVO_PIN 19

// Creating a Servo object
Servo myServo;

// Initial servo angle
int servoAngle = 90;

// Defining two flags to control the motion of the servo motor
bool servoUp = false;
bool servoDown = false;

// Defining second Servo motor pin
#define SERVO2_PIN 22

// Creating teh second Servo object
Servo myServo2;

// Initial second servo angle
int servoAngle2 = 180;

// Defining two flags to control the motion of the second servo motor
bool servoUp2 = false;
bool servoDown2 = false;

// Defining variables to implement non-blocking servo move logic
unsigned long lastServoMove = 0;
unsigned long lastServoMove2 = 0;
const unsigned long servoInterval = 15;

// Defining Ultrasonic Sensor Pins
int trigPin = 9;
int echoPin = 8;

// Defining current distance between the robot and obstacle
int currentDistance;

// Defining Distance threshold for obstacle avoidance
int distanceThreshold = 30;

// Defining a flag in case of an obstacle
bool obstacleDetected = false;

// Defining a frequency of 1000 ms or 1 second
// about distance printing
unsigned long lastDistancePrint = 0;
const unsigned long distancePrintInterval = 1000;

// Defining a frequency of 1000 ms or 1 second
// about echo occurance printing
unsigned long lastNoEchoWarning = 0;

const unsigned long noEchoWarningInterval = 1000;

// This flag is used to override the logic when
// the robot has a static obstacle in front of it
// so that it can move forward despite the obstacle
// for a purpose e.g. pushing the obstacle if it is soft or else
bool override = false;

// Defining the left IR sensor
#define leftIRPin 17

// Defining right IR sensor
#define rightIRPin 18

// Defining a variable to store the last turn of the robot
// 'L' = last turned left, 'R' = right, 'N' = none
char lastTurn = 'N';

// Defining the motion states of the robot
enum MotionState {STOPPED, MOVING_FORWARD, MOVING_BACKWARD, TURNING_LEFT, TURNING_RIGHT};

// Defining the default current motion state of the robot
MotionState currentMotion = STOPPED;

// Defining the maximum attempts the robot can take for recovery mode
const int MAXLINELOSSATTEMPTS = 10;

// Defining a counter to count the attempts
int lineLossCounter = 0;

// Defining a flag to control the permanent stop of motors
bool stopForever = false;

// Defining Left side motors (2 motors in parallel) via L298N Channel A
// IN1 Pin - Left motor backward
int motor1pin1 = 2;

// IN2 Pin - Left motor forward
int motor1pin2 = 3; 

// ENA Pin - Speed control for left motors (PWM)
int ENA = 6;         

// Defining Right side motors (2 motors in parallel) via L298N Channel B
// IN3 Pin - Right motor backward 
int motor2pin1 = 4;  

// IN4 Pin - Right motor forward 
int motor2pin2 = 5;

// ENB Pin - Speed control for right motors (PWM)
int ENB = 10;      

// Defining robot's speed
int speed;

// Defining robot's normal motor speed, range: 0–255
const int motorSpeed = 210;

// MPU object
MPU6050 mpu;

// Defining accelerometer and gyroscope variables for X, Y, Z axis
int16_t axRaw, ayRaw, azRaw, gxRaw, gyRaw, gzRaw;

// Variables to store acceleration in (g)ravity units
float axg, ayg, azg;

// Defining X, Z axis angles in a 3D world
float pitch = 0.0;
float roll = 0.0;

// Defining the filtered X, Z axis angles
float filteredPitch = 0.0;
float filteredRoll = 0.0;

// Defining alpha, a hyperparameter
// Range between: 0 (fast) and 1 (slow smoothing) 
const float alpha = 0.85;

// Defining the maximum angle in degrees 
// the robot can tilt, for safety purposes
const float MAXSAFEANGLE = 25.0;

// Defining a flag to check whenever the robot has tilted
bool robotTipped = false;

// Ensuring tilt after 620 ms or 0.5 second for real-world application 
unsigned long tiltStartTime = 0;
const unsigned long tiltDelay = 620;
unsigned long lastSlopePrint = 0;

// Defining motion detection threshold
// The robot will stop if the acceleration is greater than 2.5g
// It is like a sudden acceleration case, like e.g a collision or a push
const float SHAKETHRESHOLD = 2.5; 

// Defining free fall detection threshold
// So below that g value:0.28, the case is considered a free fall or a pickup
const float FREEFALLTHRESHOLD = 0.28;

// Defining motion event flags, in case of a sudden shake or free fall
bool shakeDetected = false;
bool freeFallDetected = false;

// Defining variables to ensure both shake and free fall
// events after a fixed time prison
unsigned long shakeStartTime = 0;
unsigned long freeFallStartTime = 0;

// Maximum time prison duration
const unsigned long SHAKECONFIRMATIONDELAY = 200;
const unsigned long FREEFALLCONFIRMATIONDELAY = 50;

// Waiting for the next new posture
// Defining posture max time prison to avoid false triggering events
unsigned long lastPostureChangeTime = 0;
const unsigned long POSTURESTABILITYDELAY = 550;

// Defining variables to store the current and last robot posture
String pendingPosture = "";
String lastPosture = "";

// Defining a posture flag used to enter
// safe mode per unsafe posture once per time
bool postureSafeMode = false;

// Defining a flag to detect unsafe posture
bool unsafePosture = false;

// Defining current posture
String newPosture;

// Defining the current terrain type as a default
String currentTerrain = "🟩 Flat terrain";

// Defining the newTerrain in case of change of terrain type
String newTerrain;
String pendingTerrain = "";

// Defining arrays used to classify the terrain type
// Number of samples to average
const int TERRAIN_DATA_SIZE = 20;  

// Acceleration data from axis X, Y, Z
float axData[TERRAIN_DATA_SIZE];
float ayData[TERRAIN_DATA_SIZE];
float azData[TERRAIN_DATA_SIZE];

// Starting point of the arrays
int bufferIndex = 0;

// Defining a flag, in case the arrays are full
bool bufferFilled = false;

// Defining the time prison of 2000 ms or 2 seconds, for security purposes
// To prevent false triggers, like in case the robot passes through a stone in stable flat terrain
// If no time prison, then the terrain type could be changed to unstable
// So by waiting 2 seconds, false trigger events are prevented   
unsigned long terrainChangeStartTime = 0;
const unsigned long terrainStabilityDelay = 2000;

// Mode for autonomous driving
bool autonomousMode = false;

// Defining a flag for safe mode activation, like stopping the motors
bool inSafeMode = false;

// milliseconds delay to debounce safe mode on/off
unsigned long lastUnsafeTime = 0;
unsigned long lastSafeTime = 0;
const unsigned long safeModeDelay = 500;

// Counting and using the current millis for non-blocking behaviour
unsigned long now;

// Defining all the required functions 
// for the robot's capabilities and behaviour

// BlackBox functions
String getTimestamp();
void   blackBox(String event);

// Motion functions
void moveForward(int speedToUse = -1);
void moveBackward(int speedToUse = -1);
void turnLeft(int speedToUse = -1);
void turnRight(int speedToUse = -1);
void stopMotors();
void setMotorSpeed(int speed);

// Environmental functions
void sendSensorData();
int  readGasSensor();

// Mode functions
void enterManualMode();
void enterAutonomousMode();
void enterSafeMode();

// Robot control commands function
void controlRobot(String command);

// IR sensors line following functions
int  measureDistance();
void lineFollowing();
bool recoverLine();

// MPU6050 functions for Robot senses
// about its pose and direction
// and for both motion and speed adjustment
// based on different scenarios like angle, terrain type, posture, sos case
int  getSpeedFromSlopeForward(float pitch);
int  getSpeedFromSlopeBackward(float pitch); 
int  getSpeedFromSlopeRightTurn(float pitch);
int  getSpeedFromSlopeLeftTurn(float pitch);
void updateTiltAngles();
void detectMotionPatterns();
void classifyPosture();
void classifyTerrain();


// This function is used to get timestamps for the robot's BlackBox
String getTimestamp() {
  
  // Calculating seconds, minutes and hours
  // for the according final timestamp
  unsigned long seconds = (now / 1000) % 60;
  
  unsigned long minutes = (now / (1000 * 60)) % 60;
  
  unsigned long hours = (now / (1000 * 60 * 60)) % 24;

  // Buffer to store timestamp
  // with a size max to 12 characters
  char buffer[12];
  
  sprintf(buffer, "%02lu:%02lu:%02lu", hours, minutes, seconds);
  
  // String format
  return String(buffer);
}


// This blackBox function, is used to log
// and store all usefull data needed
// to evaluate causes, like in case of
// a fatal crash or other important cases
// these logs/info will be sent and fethed by a txt file
// in a Python interface environment
// used as the robot's Base Station
void blackBox(String event) {
 
  Serial.print("[BLACKBOX][");
 
  Serial.print(getTimestamp());
 
  Serial.print("] ");
 
  Serial.println(event);
}


// Setup function, for initialization purposes
void setup() {

  // Initializing Serial Monitor
  Serial.begin(9600);

  // Letting sensors stabilize after power-up 
  delay(2500);

  // Initializing the DHT11 sensor
  dht.begin();

  // Attach the servo motor
  myServo.attach(SERVO_PIN);

  // Set to initial position
  myServo.write(servoAngle);

  // Attach the second servo motor
  myServo2.attach(SERVO2_PIN);

  // Set to initial position
  myServo2.write(servoAngle2);

  // Motor Pins Setup
  pinMode(motor1pin1, OUTPUT);
  pinMode(motor1pin2, OUTPUT);
  
  pinMode(motor2pin1, OUTPUT);
  pinMode(motor2pin2, OUTPUT);
  
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);

  // Ultrasonic Pins setup
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  // IR Pins setup
  pinMode(leftIRPin, INPUT);
  pinMode(rightIRPin, INPUT);

  // I2C for Arduino Mega microcontroller
  // is on pins 20 (SDA) and 21 (SCL)
  Wire.begin();
  blackBox("Initializing MPU6050 ...");
  mpu.initialize();

  // Checking mpu connection attempt
  if (mpu.testConnection()) {
    
    blackBox("✅ MPU6050 connection successful!");
  }
  
  else {
  
    blackBox("❌ MPU6050 connection failed!");
  
    while (1);
  }

  blackBox("Bluetooth connection Ready!");

  // For safety purposes, all motors are stopped on startup
  stopMotors();  
}


// Loop function, for the actual running process
void loop() {
  
  // Millis counting
  now = millis();

  // Checking if Bluetooth has sent any data 
  if (Serial.available()) {
    
    // Taking the command sent by the user via Python PyGame UI
    String command = Serial.readStringUntil('\n');
    
    // Trimming the string content
    command.trim();

    // Converting command to lowercase
    command.toLowerCase();

    // Implementing the servo movement based on the sent commands
    if (command == "servo_up_start") {
      
      servoUp = true;
      servoDown = false;
      
      blackBox("Servo moving up started");
      
      return;
    }

    else if (command == "servo_up_stop") {
    
      servoUp = false;
    
      blackBox("Servo moving up stopped");
    
      return;
    }

    else if (command == "servo_down_start") {
    
      servoDown = true;
      servoUp = false;
    
      blackBox("Servo moving down started");
    
      return;
    }

    else if (command == "servo_down_stop") {
    
      servoDown = false;
    
      blackBox("Servo moving down stopped");
    
      return;
    }

    // Servo command handler for angle
    if (command.startsWith("servo:")) {
      
      // Taking the angle sent by the user
      int angle = command.substring(6).toInt();

      // Checking the range for correctness
      if (angle >= 0 && angle <= 180) {
    
        servoAngle = angle;

        // Resetting flags to avoid conflicts
        servoUp = false;
        servoDown = false;
    
        blackBox("✅ Servo angle set to: " + servoAngle);
      }

      else {
      
        blackBox("⚠️ Invalid angle! Must be between 0 and 180.");
      }
      
      return;
    }

    if (command == "servo2_up_start") {
      
      servoUp2 = true;
      servoDown2 = false;
      
      blackBox("Servo 2 moving up started");
      
      return;
    }
    
    else if (command == "servo2_up_stop") {
      
      servoUp2 = false;
      
      blackBox("Servo 2 moving up stopped");
      
      return;
    }
   
    else if (command == "servo2_down_start") {
      
      servoDown2 = true;
      servoUp2 = false;
      
      blackBox("Servo 2 moving down started");
      
      return;
    }
    
    else if (command == "servo2_down_stop") {
    
      servoDown2 = false;
     
      blackBox("Servo 2 moving down stopped");
     
      return;
    }
    
    else if (command.startsWith("servo2:")) {
      
      int angle = command.substring(7).toInt();
      
      if (angle >= 0 && angle <= 180) {
      
        servoAngle2 = angle;
        servoUp2 = false;
        servoDown2 = false;
        
        blackBox("✅ Servo 2 angle set to: " + String(servoAngle2));
      }
    
      else {
      
        blackBox("⚠️ Invalid angle for Servo 2! Must be 0–180.");
      }
    
      return;
    }

    // Speed change commands ignored, fixed speed used
    if (command.startsWith("speed")) {
       
        Serial.println("⚠️ Speed change command ignored: fixed speed is used.");
       
        return;
    }

    // Only remote mode commands now
    controlRobot(command);

    delay(20);
  }

  // Updating servoAngle target based on servoUp/servoDown
  // With step 3
  if (servoUp) {
  
    servoAngle = min(servoAngle + 3, 180);
  }
  
  else if (servoDown) {
    
    servoAngle = max(servoAngle - 3, 0);
  }

  // Moving servo one step toward servoAngle at intervals
  if (now - lastServoMove > servoInterval) {
  
    int currentPos = myServo.read();

    if (currentPos < servoAngle) {
      
      myServo.write(currentPos + 3);
      
      lastServoMove = now;
    }

    else if (currentPos > servoAngle) {
      
      myServo.write(currentPos - 3);
      
      lastServoMove = now;
    }
  
    else {
      
      // Reached target angle, resetting flags
      servoUp = false;
      servoDown = false;
    }
  }

    // Update servoAngle2 based on flags
  if (servoUp2) {
    
    servoAngle2 = min(servoAngle2 + 3, 180);
  }
  
  else if (servoDown2) {
  
    servoAngle2 = max(servoAngle2 - 3, 0);
  }

  // Move servo2 step by step
  if (now - lastServoMove2 > servoInterval) {
  
    int currentPos2 = myServo2.read();
  
    if (currentPos2 < servoAngle2) {
  
      myServo2.write(currentPos2 + 3);
  
      lastServoMove2 = now;
    }
    
    else if (currentPos2 > servoAngle2) {
    
      myServo2.write(currentPos2 - 3);
    
      lastServoMove2 = now;
    }
    
    else {
    
      servoUp2 = false;
      servoDown2 = false;
    }
  }

  // Sending sensor data via Bluetooth every 5 seconds
  if (now - lastDHT11Read >= DHT11FREQUENCY) {
      
      lastDHT11Read = now;
      
      sendSensorData();
  }

  // Taking the current distance
  currentDistance = measureDistance();

  // In case that the distance is less or equal to the distance threshold
  // then the robot will stop until the path is clear again  
  if (now - lastDistancePrint >= distancePrintInterval) {
  
      lastDistancePrint = now;
  
      blackBox("Distance: " + String(currentDistance));
  }
  
  // In case that the robot is in autonomous mode 
  if (autonomousMode) {

    // In case the robot is stopped forever due to several line detection failed attempts
    if (stopForever) {

      stopMotors();

      return;
    }
    
    // In autonomous mode still
    // In case the robot has in front of it an obstacle in a distance of <= 30cm
    // The robot will stop
    if (currentDistance <= distanceThreshold) { 
    
        obstacleDetected = true;
    
        stopMotors();
    
        delay(20);
    }
    
    // Otherwise, the robot will continue moving
    else {
    
        obstacleDetected = false;
        
        // Autonomous mode runs line following function
        lineFollowing();
    }
  }
  
  // Else in case the robot is in manual mode, same logic about the distance <= 30
  else {
  
      obstacleDetected = (currentDistance <= distanceThreshold);
  }

  // If the robot is in manual mode, has an obstacle <= 30cm
  // is moving forward and there is no override, the robot will stop moving forward only
  // It can move backward, left and right to overcome the obstacle
  if (!autonomousMode && obstacleDetected && currentMotion == MOVING_FORWARD && !override) {
      
      stopMotors();
      
      currentMotion = STOPPED;
      
      blackBox("🚫 Obstacle detected! Stopping motors immediately (manual mode).");
  }

  // Performing tilt function
  updateTiltAngles();

  // Implementing speed adjustment based on the angle
  // [Uphill, Downhill or stable] and movement is forward
  if (currentMotion == MOVING_FORWARD) {
   
    speed = getSpeedFromSlopeForward(filteredPitch);
    
    moveForward(speed);
  }

  // Implementing speed adjustment based on the angle
  // [Uphill, Downhill or stable] and movement is backward
  else if (currentMotion == MOVING_BACKWARD) {
  
    speed = getSpeedFromSlopeBackward(filteredPitch);
    
    moveBackward(speed);
  }

  // Implementing speed adjustment based on the angle
  // [Uphill, Downhill or stable] and movement is right turn
  if (currentMotion == TURNING_RIGHT) {
   
    speed = getSpeedFromSlopeRightTurn(filteredPitch);
    
    turnRight(speed);
  }

  // Implementing speed adjustment based on the angle
  // [Uphill, Downhill or stable] and movement is left turn
  else if (currentMotion == TURNING_LEFT) {
  
    speed = getSpeedFromSlopeLeftTurn(filteredPitch);
    
    turnLeft(speed);
  }

  // a delay of 20 milliseconds or 0.02 seconds per loop
  delay(20);
}


// Function to move forward
void moveForward(int speedToUse = -1) {
  
  // In case no speed value was given
  // speed will take the default value of 210
  if (speedToUse == -1) {
    
    speedToUse = motorSpeed;
  }

  // Ensuring speed is between 0 and 255, correct range
  speedToUse = constrain(speedToUse, 0, 255);

  // Inserting the given controlled speed
  setMotorSpeed(speedToUse);

  // Forward motion logic below
  digitalWrite(motor1pin1, LOW);
  digitalWrite(motor1pin2, HIGH);

  digitalWrite(motor2pin1, LOW);
  digitalWrite(motor2pin2, HIGH);
}


// Function to move backward
void moveBackward(int speedToUse = -1) {

  // In case no speed value was given
  // speed will take the default value of 210
  if (speedToUse == -1) {
    
    speedToUse = motorSpeed;
  }

  // Ensuring speed is between 0 and 255, correct range
  speedToUse = constrain(speedToUse, 0, 255);
  
  // Inserting the given controlled speed
  setMotorSpeed(speedToUse);

  // Backward motion logic below
  digitalWrite(motor1pin1, HIGH);
  digitalWrite(motor1pin2, LOW);

  digitalWrite(motor2pin1, HIGH);
  digitalWrite(motor2pin2, LOW);
}


// Function to turn left
void turnLeft(int speedToUse = -1) {
  
  // In case no speed value was given
  // speed will take the default value of 210
  if (speedToUse == -1) {
    
    speedToUse = motorSpeed;
  }
  
  // Ensuring speed is between 0 and 255, correct range
  speedToUse = constrain(speedToUse, 0, 255);
  
  // Inserting the given controlled speed
  setMotorSpeed(speedToUse);

  // Turning left motion logic below
  digitalWrite(motor1pin1, HIGH);
  digitalWrite(motor1pin2, LOW);

  digitalWrite(motor2pin1, LOW);
  digitalWrite(motor2pin2, HIGH);
}


// Function to turn right
void turnRight(int speedToUse = -1) {

  // In case no speed value was given
  // speed will take the default value of 210
  if (speedToUse == -1) {
    
    speedToUse = motorSpeed;
  }
  
  // Ensuring speed is between 0 and 255, correct range
  speedToUse = constrain(speedToUse, 0, 255);

  // Inserting the given controlled speed
  setMotorSpeed(speedToUse);

  // Turning right motion logic below
  digitalWrite(motor1pin1, LOW);
  digitalWrite(motor1pin2, HIGH);

  digitalWrite(motor2pin1, HIGH);
  digitalWrite(motor2pin2, LOW);
}


// Function to stop motors
void stopMotors() { 
  
  // Setting motor speed to 0
  setMotorSpeed(0);

  // Stopping motors motion logic below
  digitalWrite(motor1pin1, LOW);
  digitalWrite(motor1pin2, LOW);

  digitalWrite(motor2pin1, LOW);
  digitalWrite(motor2pin2, LOW);  
}


// This function is used to set motor speed PWM 
void setMotorSpeed(int speed) {
  
  // Constraining speed between 0 and 255, correct range
  speed = constrain(speed, 0, 255);

  // Applying PWM speed to motors
  analogWrite(ENA, speed);
  analogWrite(ENB, speed);
}


// Function to send sensor Temperature, Humidity and Gas
// data via Bluetooth in a json format
void sendSensorData() {
    
    // Defining a Json object: environData from environmental data
    StaticJsonDocument<200> environData;

    // Defining a nested JSON object for Gas
    JsonObject gasObj = environData.createNestedObject("Gas");

    // Defining a variable to store the content from json format
    String jsonString;

    // Defining the status of the air condition based on gas value
    String gasStatus;

    // Reading temperature values as Celsius
    float tempValue = dht.readTemperature();
    
    // Reading humidity values
    float humValue = dht.readHumidity();

    // Checking if any reads failed
    // Case of failure, value -1
    if (isnan(tempValue) || isnan(humValue)) {
    
        blackBox("⚠️ Failed to read from DHT11 sensor!");
        
        environData["Temperature"] = -1;
        environData["Humidity"] = -1;
    }
    
    // Case of success
    else {
    
        environData["Temperature"] = tempValue;
        environData["Humidity"] = humValue;
    }

    // Reading gas values
    int gasValue = readGasSensor();

    // Checking if any reads failed
    // Case of failure, value -1
    if (isnan(gasValue)) {
    
        blackBox("⚠️ Failed to read from Gas sensor!");
        
        gasObj["value"] = -1;
        gasObj["status"] = "Unknown";
    }
    
    // Case of success
    else {

      // Case of excellent condition
      if (gasValue < 100) {
      
          gasStatus = "✅ Very clean air";
      }
      
      // Case of normal condition
      else if (gasValue < 200) {
      
          gasStatus = "🟢 Normal indoor air";
      }
      
      // Case of semi emergency condition
      else if (gasValue < 300) {
      
          gasStatus = "🟡 Ventilation suggested";
      }
      
      // Case of emergency condition
      else if (gasValue < 500) {
        
          gasStatus = "🟠 Risk of health effects";
      }
      
      // Case of super emergency condition
      else {
      
          gasStatus = "⚠️ Dangerous!";
      }
      
      gasObj["value"] = gasValue;
      gasObj["status"] = gasStatus;
    }

    serializeJson(environData, jsonString);
    
    Serial.println((jsonString));
  }


// Function to read and return analog sensor data (gas sensor)
int readGasSensor() {

  // Taking the gas data
  unsigned int sensorValue = analogRead(GAS_PIN);

  return sensorValue;
}


// This function is used to reset the flags when manual mode
void enterManualMode() {
  
  // If the robot is in safe mode, thus stopped
  // Now will be able to move again
  if (inSafeMode) {
    
    // Disabling safe mode 
    inSafeMode = false;
    
    blackBox("🟢 Safe mode manually exited.");
    
    return;  
  }

  // Resetting autonomous mode to false
  // Since, now, manual mode will be in power
  autonomousMode = false;

  // Resetting override flag
  override = false;
  
  blackBox("Override DISABLED due to manual mode.");

  // Resetting the permanent stop of motors to false
  stopForever = false;
  
  // Resetting smart recovery to None
  lastTurn = 'N';

  // Setting the current movement to none
  currentMotion = STOPPED;
  
  // Stopping motors to initiate manual mode
  stopMotors();
  
  blackBox("🕹️ Manual mode enabled");
}


// This function is used to reset the flags when autonomous mode
void enterAutonomousMode() {
  
  // The robot will not be able to enter autonomous mode
  // if it is in safe mode, thus stopped
  if (inSafeMode) {
  
    blackBox("⚠️ Cannot enter autonomous mode while in safe mode!");
  
    return;
  }

  // Starting autonomous navigation
  autonomousMode = true;

  // Resetting the permanent stop of motors to false
  stopForever = false;

  // Resetting smart recovery to None
  lastTurn = 'N';

  // Setting the current movement to none
  currentMotion = STOPPED;

  // Safe start
  stopMotors();
  
  blackBox("🤖 Autonomous mode enabled");
}


// This function is used to enter robot
// into a safe mode when it is in danger positions
void enterSafeMode() {
  
  // if its not already into a safe mode
  // Then will enter now
  if (!inSafeMode) {
    
    // Activation of safe mode
    inSafeMode = true;

    // Stopping the motors until safe mode is false
    stopMotors();

    currentMotion = STOPPED;

    blackBox("⚠️ SAFE MODE ENABLED due to unsafe condition.");
  }
}


// Function to control robot remotely
void controlRobot(String command) {

  blackBox("DEBUG: Command received: " + command);

  // Checking which command is sent
  // and performing the according actions

  // Case override
  if (command == "o") {

    override = true;

    blackBox("⚠️ Obstacle avoidance override ENABLED!");

    return;
  }

  // Case to quit override
  else if (command == "q") {

    override = false;

    blackBox("Override DISABLED!");

    return;
  }

  // Ignoring all commands except "m" if in safe mode
  // to exit safe mode
  if (inSafeMode && command != "m") {
    
    blackBox("🛑 Robot in SAFE MODE. Send 'm' to exit safe mode.");
    
    stopMotors();
    
    currentMotion = STOPPED;
    
    return;
  }

  // Case of an obstacle <= 30cm and no override, forward command is denied
  if ((command == "f") && obstacleDetected && !override) {
    
    blackBox("🚫 Obstacle detected! Forward command ignored.");
    
    stopMotors();
    
    currentMotion = STOPPED;
    
    return;
  }

  // Case of an obstacle <= 30cm and override, forward command is performed
  else if ((command == "f") && obstacleDetected && override) {
    
    // Adjusting speed based on the terrain slope and type
    speed = getSpeedFromSlopeForward(filteredPitch);
    
    moveForward(speed);
    currentMotion = MOVING_FORWARD;

    blackBox("⚠️ Override: Moving forward despite obstacle. Adaptive speed: ");
  }

  // Case to move forward
  else if (command == "f") {

    // Adjusting speed based on the terrain slope and type
    speed = getSpeedFromSlopeForward(filteredPitch);

    moveForward(speed);
    currentMotion = MOVING_FORWARD;

    blackBox("🟢 Moving forward with adaptive speed: " + String(speed));
  }

  // Case to move backward
  else if (command == "b") {  
    
    // Adjusting speed based on the terrain slope and type
    speed = getSpeedFromSlopeBackward(filteredPitch);

    moveBackward(speed);
    currentMotion = MOVING_BACKWARD;

    blackBox("🔵 Moving backward with adaptive speed: " + String(speed));
  }
 
 // Case to turn left
  else if (command == "l") {
    
    // Adjusting speed based on the terrain slope and type
    speed = getSpeedFromSlopeLeftTurn(filteredPitch);
    
    turnLeft(speed);
    currentMotion = TURNING_LEFT;

    blackBox("Turning left with adaptive speed: " + String(speed));
  }
  
  // Case to turn right
  else if (command == "r") {
  
    // Adjusting speed based on the terrain slope and type
    speed = getSpeedFromSlopeRightTurn(filteredPitch);
    
    turnRight(speed);
    currentMotion = TURNING_RIGHT;

    blackBox("Turning right with adaptive speed: " + String(speed));
  }
  
  // Case to stop the robot
  else if (command == "s") {
  
      currentMotion = STOPPED;
   
      stopMotors();
  }

  // Case for manual movement
  else if (command == "m") {
    
    // Manual mode activation
    enterManualMode();
  }

  // Case for autonomous movement
  else if (command == "a") {
    
    autonomousMode = true;
    
    blackBox("🧠 Autonomous mode enabled");
  }
  
  // Case of invalid command
  else {
  
    blackBox("Invalid command! Try again.");
  }
}


// Function that measures the distance between the robot and the obstacle
int measureDistance() {
    
  // Ensuring the trigger pin is low before starting
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  
  // Sending a 10-microsecond pulse to trigger the sensor
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  
  digitalWrite(trigPin, LOW);

  // Listening for the echo
  long duration = pulseIn(echoPin, HIGH, 30000); // 30 ms timeout

  if (duration == 0) {
       
    if (millis() - lastNoEchoWarning > noEchoWarningInterval) {
      
      // No echo received — treated as no obstacle or very far away
      blackBox("⚠️ No echo received from ultrasonic sensor! No obstacle detected or very far away.");
      
      lastNoEchoWarning = millis();
    }
    
    return 999; 
  }

  // Calculating distance (speed of sound = 0.034 cm/µs)
  int distance = duration * 0.034 / 2;

  // Returning the distance between the robot and the obstacle
  return distance;
}


// This function serves for line following using the two IRs sensors
void lineFollowing() {
  
  // Reading IR sensor values
  // 1 = black, 0 = white
  int left = digitalRead(leftIRPin);   
  int right = digitalRead(rightIRPin); 

  // Printing binary sensor state for debugging
  Serial.print("Line Sensor Readings - Left: ");
  Serial.print(left);
  
  Serial.print(" | Right: ");
  Serial.println(right);

  // Case both sensors see black line
  if (left == 1 && right == 1) {
    
    // Adjusting the speed based on slope and type of the terrain 
    speed = getSpeedFromSlopeForward(filteredPitch);
    moveForward(speed);
    
    currentMotion = MOVING_FORWARD;
    lastTurn = 'N';

    Serial.print("Line following with adaptive speed: ");

    return;
  }

  // Case: only left sees black
  else if (left == 1 && right == 0) {
  
    // Adjusting the speed based on slope and type of the terrain 
    speed = getSpeedFromSlopeLeftTurn(filteredPitch);
    turnLeft(speed);

    currentMotion = TURNING_LEFT;
    lastTurn = 'L';

    Serial.print("Line following with adaptive speed: ");

    return ;
  }

  // Case: only right sees black
  else if (left == 0 && right == 1) {
  
    // Adjusting the speed based on slope and type of the terrain 
    speed = getSpeedFromSlopeRightTurn(filteredPitch);
    turnRight(speed);
    
    currentMotion = TURNING_RIGHT;
    lastTurn = 'R';
    
    Serial.print("Line following with adaptive speed: ");

    return;
  }

  // Case: both off the line (white surface)
  else if (left == 0 && right == 0) {
  
    // stopMotors();
    
    // currentMotion = STOPPED;
  
    // Serial.println("⚠️ Line lost! Both sensors = 0");
  

    Serial.println("⚠️ Line lost! Searching Mode Activation.");

    // Checking if the robot is realigned
    if (!recoverLine()) {
      
      // Times of no success
      lineLossCounter++;
      
      Serial.print("❌ Recovery failed. Total failures: ");
      
      Serial.println(lineLossCounter);

      stopMotors();
      
      currentMotion = STOPPED;

      // Case where there are many failures
      if (lineLossCounter >= MAXLINELOSSATTEMPTS) {
      
        Serial.println("🛑 Too many line loss events. Stopping permanently.");

        stopForever = true;

        // Stopping forever
        stopMotors(); 

        return;
      }
    }
    
    // Case where the robot found the line successfuly
    else {
    
      Serial.println("✅ Line recovery successful!");

      // Resetting after successful recovery
      lineLossCounter = 0;  
    }
  }
}


// This function is for line recovery 
bool recoverLine() {
  
  // Trying 6 times with increasing durations
  const int maxAttempts = 6;
    
  // Trying different durations, to access different angles 
  int durations[maxAttempts] = {150, 200, 250, 300, 350, 400};

  // Trying to reallign with different scenarios
  for (int i = 0; i < maxAttempts; i++) {
    
    // Starting counting millis
    unsigned long lastTime = millis();

    // Checking which was the last turn when the robot lost its track
    // And then performing the measures

    // Case where the track was lost by drifting left side
    if (lastTurn == 'L') {
      
      // Adjusting the speed based on slope and type of the terrain 
      speed = getSpeedFromSlopeRightTurn(filteredPitch);
      turnRight(speed);
    }
    
    // Case where the track was lost by drifting right side
    else if (lastTurn == 'R') {
    
      // Adjusting the speed based on slope and type of the terrain 
      speed = getSpeedFromSlopeLeftTurn(filteredPitch);
      turnLeft(speed);
    }
    
    // default direction
    else {
    
      // Adjusting the speed based on slope and type of the terrain 
      speed = getSpeedFromSlopeLeftTurn(filteredPitch);
      turnLeft(speed);
    }

    // Waiting during turning
    // Non-blocking wait logic with small delay to reduce CPU load
    unsigned long nowLocal;
    
    do {
    
      nowLocal = millis();
      
      // Small delay to avoid CPU load
      delay(1);
    
    } while (nowLocal - lastTime < durations[i]);

    // Stopping motors to check for the black line
    stopMotors();

    // Updating sensor readings
    int left = digitalRead(leftIRPin);
    int right = digitalRead(rightIRPin);

    // Left recovery
    if (lastTurn == 'L' && left == 1 && right == 0) {

      // Moving forward until left ir is 1 again
      while (digitalRead(leftIRPin) != 1) {
        
        // Adjusting the speed based on slope and type of the terrain
        speed = getSpeedFromSlopeForward(filteredPitch);
        
        moveForward(speed);
      }

      stopMotors();
      
      // Adjusting the speed based on slope and type of the terrain 
      speed = getSpeedFromSlopeLeftTurn(filteredPitch);
      
      turnLeft(speed);
      
      lastTurn = 'L';
      
      return true;
    }

    // Right recovery
    if (lastTurn == 'R' && right == 1 && left == 0) {
      
      // Moving forward until right ir is 1 again
      while (digitalRead(rightIRPin) != 1) {
        
        // Adjusting the speed based on slope and type of the terrain
        speed = getSpeedFromSlopeForward(filteredPitch);
        
        moveForward(speed);
      }

      stopMotors();
      
      // Adjusting the speed based on slope and type of the terrain 
      speed = getSpeedFromSlopeRightTurn(filteredPitch);
      
      turnRight(speed);
      
      lastTurn = 'R';
      
      return true;
    }

  }

  // Case where it failed after all attempts
  return false; 
}


// This function is a math formula, to adjust speed based
// on mapping ground roughness and ground degrees [uphill, downhill]
float mapSpeed(float xPitch, float minDeg, float maxDeg, float minStep, float maxStep) {
  
  // Avoiding division by zero
  if (minDeg == maxDeg) {
    
    return minStep;
  }

  // Ensuring no speed jumps or unpredictable
  // behavior due to extrapolation, like out of ranges angles [in degrees]
  if (xPitch < minDeg) {
  
    xPitch = minDeg;
  }
  
  if (xPitch > maxDeg) {
  
    xPitch = maxDeg;
  }
  
  return (xPitch - minDeg) * (maxStep - minStep) / (maxDeg - minDeg) + minStep;
}


// Function to adjust motor speed based on slope angle
// and terrain type for forward movement
int getSpeedFromSlopeForward(float pitch) {
  
  // Used for base speed calculation
  float terrainWeight;
  int maxBaseSpeed = 210;

  // Assigning terrain weights
  if (currentTerrain == "🟩 Flat terrain") {
    
    terrainWeight = 1.0; // 210 speed
  }
  
  else if (currentTerrain == "🟨 Medium rough terrain") {
  
    terrainWeight = 0.857; // ~180 speed
  }
  
  else if (currentTerrain == "🟥 Rough terrain!") {
   
    terrainWeight = 0.762; // ~160 speed
  }
  
  else {

    // Backup value in case of unknown terrain
    // Setting it low for security, in case
    // of a super rough terrain, dangerous
    // for the robot's safety
    terrainWeight =  0.667; // ~140 speed  
    
    blackBox("⚠️ Unknown terrain type!");
  }

  // Base speed influenced by terrain
  int baseSpeed = maxBaseSpeed * terrainWeight;

  // Dynamic pitch adjustment 
  float pitchAdjustment = 0.0;

  // Range of uphill: 11 to 24 degrees
  if (pitch >= 11 && pitch <= 24.99) {
    
    pitchAdjustment = mapSpeed(pitch, 11, 24.99, 10, 30);
  }

  // Range of downhill: -11 to -24 degrees
  else if (pitch <= -11 && pitch >= -24.99) {
  
    // pitch is negative, so using absolute value for mapping
    // Dowhill case
    float absPitch = abs(pitch);

    pitchAdjustment = -mapSpeed(absPitch, 11, 24.99, 13.3, 39.9);
  }

  // Final adaptive speed after mapSpeed formula
  int finalSpeed = baseSpeed + pitchAdjustment;

  // Returning the final speed and
  // ensuring its range of speed: 130 to 255
  return constrain(finalSpeed, 130, 255);
}


// Function to adjust motor speed based on slope angle
// and terrain type for backward movement
int getSpeedFromSlopeBackward(float pitch) {
  
  // Used for base speed calculation
  float terrainWeight;
  int maxBaseSpeed = 210;

  // Assigning terrain weights
  if (currentTerrain == "🟩 Flat terrain") {
    
    terrainWeight = 1.0; // 210 speed
  }
  
  else if (currentTerrain == "🟨 Medium rough terrain") {
  
    terrainWeight = 0.857; // ~180 speed
  }
  
  else if (currentTerrain == "🟥 Rough terrain!") {
   
    terrainWeight = 0.762; // ~160 speed
  }
  
  else {

    // Backup value in case of unknown terrain
    // Setting it low for security, in case
    // of a super rough terrain, dangerous
    // for the robot's safety
    terrainWeight =  0.667; // ~140 speed  
    
    blackBox("⚠️ Unknown terrain type!");
  }

  // Base speed influenced by terrain
  int baseSpeed = maxBaseSpeed * terrainWeight;

  // Dynamic pitch adjustment for backward logic
  // that is inverse of forward logic
  float pitchAdjustment = 0.0;

  // Uphill when going backward means pitch is positive,
  // so reduce speed smoothly between 11 to 24 degrees
  if (pitch >= 11 && pitch <= 24.99) {
    
    // Uphill reduces speed by 13.3 to 39.9 points progressively
    pitchAdjustment = -mapSpeed(pitch, 11, 24.99, 13.3, 39.9);
  }
  
  // Downhill when going backward means pitch is negative,
  // so increase speed smoothly between -11 to -24 degrees
  else if (pitch <= -11 && pitch >= -24.99) {
    
    // Downhill increases speed by 10 to 30 points progressively
    float absPitch = abs(pitch);
   
    pitchAdjustment = mapSpeed(absPitch, 11, 24.99, 10, 30);
  }

  // Final adjusted speed combining terrain and pitch effects
  int finalSpeed = baseSpeed + pitchAdjustment;

  // Ensuring PWM limits: 130 to 255
  return constrain(finalSpeed, 130, 255);
}


// Function to adjust motor speed based on slope angle
// and terrain type for right movement
int getSpeedFromSlopeRightTurn(float pitch) {
  
  // Used for base speed calculation
  float terrainWeight;
  int maxBaseSpeed = 190; 

  // Assigning terrain weights to match desired turn speeds
  if (currentTerrain == "🟩 Flat terrain") {
    
    terrainWeight = 0.85; // ~162
  }
  
  else if (currentTerrain == "🟨 Medium rough terrain") {
  
    terrainWeight = 0.9; // ~171
  }
  
  else if (currentTerrain == "🟥 Rough terrain!") {
  
    terrainWeight = 1.0; // ~190
  }
  
  else {
    
    // In this case where the ground is too rough
    // is given to robot a turning speed power enough to
    // not damage itself but hopefully escape as well
    terrainWeight = 0.8; // 152
  
    blackBox("⚠️ Unknown terrain type!");
  }

  // Base speed influenced by terrain
  int baseSpeed = maxBaseSpeed * terrainWeight;

  // Dynamic pitch adjustment
  float pitchAdjustment = 0.0;

  // Uphill and Downhill turning
  // reducing turning speed to avoid tipping
  if (pitch >= 11 && pitch <= 24.99) {
    
    // Uphill turning reduces speed progressively 13.3 to 39.9
    pitchAdjustment = -mapSpeed(pitch, 11, 24.99, 13.3, 39.9);
  } 

  else if (pitch <= -11 && pitch >= -24.99) {
   
    // Downhill turning reduces speed progressively 13.3 to 39.9
    float absPitch = abs(pitch);
   
    pitchAdjustment = -mapSpeed(absPitch, 11, 24.99, 13.3, 39.9);
  }

  // Final speed after terrain and pitch effect
  int finalSpeed = baseSpeed + pitchAdjustment;

  // Constraining speed range for safety
  return constrain(finalSpeed, 140, 200);
}


// Function to adjust motor speed based on slope angle
// and terrain type for left movement
int getSpeedFromSlopeLeftTurn(float pitch) {
  
  // Used for base speed calculation
  float terrainWeight;
  int maxBaseSpeed = 190; 

  // Assigning terrain weights to match desired turn speeds
  if (currentTerrain == "🟩 Flat terrain") {
    
    terrainWeight = 0.85; // ~162
  }
  
  else if (currentTerrain == "🟨 Medium rough terrain") {
  
    terrainWeight = 0.9; // ~171
  }
  
  else if (currentTerrain == "🟥 Rough terrain!") {
  
    terrainWeight = 1.0; // ~190
  }
  
  else {
    
    // In this case where the ground is too rough
    // is given to robot a turning speed power enough to
    // not damage itself but hopefully escape as well
    terrainWeight = 0.8; // 152
  
    blackBox("⚠️ Unknown terrain type!");
  }

  // Base speed influenced by terrain
  int baseSpeed = maxBaseSpeed * terrainWeight;

  // Dynamic pitch adjustment
  float pitchAdjustment = 0.0;

  // Uphill and Downhill turning
  // reducing turning speed to avoid tipping
  if (pitch >= 11 && pitch <= 24.99) {
    
    // Uphill turning reduces speed progressively 13.3 to 39.9
    pitchAdjustment = -mapSpeed(pitch, 11, 24.99, 13.3, 39.9);
  } 

  else if (pitch <= -11 && pitch >= -24.99) {
   
    // Downhill turning reduces speed progressively 13.3 to 39.9
    float absPitch = abs(pitch);
   
    pitchAdjustment = -mapSpeed(absPitch, 11, 24.99, 13.3, 39.9);
  }

  // Final speed after terrain and pitch effect
  int finalSpeed = baseSpeed + pitchAdjustment;

  // Constraining speed range for safety
  return constrain(finalSpeed, 140, 200);
}


// This is used for tilt detection, if ≥ 25 degrees, danger!
void updateTiltAngles() {
  
  // Getting raw acceleration and gyroscope data from X, Y, Z axis
  mpu.getMotion6(&axRaw, &ayRaw, &azRaw, &gxRaw, &gyRaw, &gzRaw);

  // Converting to g units
  // Accelerometer raw values range from –32768 to 32767, for ±2g
  // 16384.0 is the number of raw units per 1g when using ±2g sensitivity
  axg = axRaw / 16384.0;
  ayg = ayRaw / 16384.0;
  azg = azRaw / 16384.0;

  // Calculating pitch and roll
  // pitch: how much the robot is tilted forward/backward
  // roll: how much it's tilted left/right
  pitch = atan2(axg, sqrt(ayg * ayg + azg * azg)) * 180 / PI;
  roll  = atan2(ayg, sqrt(axg * axg + azg * azg)) * 180 / PI;

  // Applying low-pass filter
  // alpha is between 0.0 and 1.0 
  // to smooth out sudden spikes, making pitch and roll more stable
  filteredPitch = alpha * filteredPitch + (1 - alpha) * pitch;
  filteredRoll  = alpha * filteredRoll +  (1 - alpha) * roll;

  // Checking for dangerous tilt
  if (abs(filteredPitch) > MAXSAFEANGLE || abs(filteredRoll) > MAXSAFEANGLE) {

    // Starting timer if not already started
    if (tiltStartTime == 0) {
     
      tiltStartTime = millis();
    }

    // If tilt persists longer than delay, triggering safe mode
    if (now - tiltStartTime >= tiltDelay && !robotTipped) {
     
      blackBox("⚠️ WARNING: Robot tipped over!");
     
      robotTipped = true;
     
      enterSafeMode();
    }
  } 
  
  else {
    
    // Resetting timer and flag when tilt is back to normal
    tiltStartTime = 0;
    
    robotTipped = false;
  }

  // Running shake/free fall detection
  // posture and terrain classification
  detectMotionPatterns();
  classifyPosture();
  classifyTerrain();
  
  // Checking the adjusted speed every 1000 ms or 1 second 
  if (now - lastSlopePrint >= 1000) {
    
    lastSlopePrint = millis();
    
    blackBox("Slope: " + String(filteredPitch, 2) + " Deg | Forward Speed: "
              + String(getSpeedFromSlopeForward(filteredPitch))
              + " | Backward Speed: " + String(getSpeedFromSlopeBackward(filteredPitch)));
  }
}


// This function is used to detect suspicious motion patterns
void detectMotionPatterns() {
  
  // Calculating acceleration magnitude (vector length)
  float acceMagnitude = sqrt(axg * axg + ayg * ayg + azg * azg);

  // Detecting sudden shakes (impact)
  if (acceMagnitude > SHAKETHRESHOLD) {

    if (shakeStartTime == 0) {
      
      // Starting timer
      shakeStartTime = now;
    }

    // Ensuring shake or collision after a period of time
    if ((now - shakeStartTime >= SHAKECONFIRMATIONDELAY) && !shakeDetected) {
      
      blackBox("🚨 Confirmed: Shake/Impact detected!");
      
      shakeDetected = true;
      
      enterSafeMode();
    }
  }
  
  else {
    
    // Resetting if no shake
    shakeStartTime = 0; 

    // Resetting flag if back to normal
    shakeDetected = false;
  }

  // Detecting pickup or free fall (very low vertical accel)
  // Usually gravity = 1g on Z-axis, so near zero means falling or picked up
  if (abs(azg) < FREEFALLTHRESHOLD) {
    
    if (freeFallStartTime == 0) {
        
        // Starting timer
        freeFallStartTime = now;
    }

    // Ensuring free fall after a period of time
    if ((now - freeFallStartTime) >= FREEFALLCONFIRMATIONDELAY && !freeFallDetected) {
  
      blackBox("⚠️ Free fall detected!");
  
      freeFallDetected = true;

      enterSafeMode();
    }
  }
  
  else {
    
    // Resetting if no free fall
    freeFallStartTime = 0;

    // Resetting flag if back to normal
    freeFallDetected = false;
  }
}


// Function to classify robot posture / Pose estimation based on MPU6050 data
void classifyPosture() {

  // Threshold for confidence
  // close to 1g on the dominant axis
  // classifying posture only when one axis sees at least 60% of gravity
  // meaning the robot is mostly still and in a clear orientation
  const float GTHRESHOLD = 0.60;

  // dominant axis must be at least 0.20g stronger than the others
  // because two axes might have similar values
  // only picking a posture if it's clearly dominant
  const float DOMINANT = 0.20; 

  // Process of finding the dominant axis, thus X, Y or Z
  // ensuring the axis is strong enough
  // and it is clearly stronger than the other two axes
  bool axDominant = (abs(axg) > GTHRESHOLD) &&
                    (abs(axg) > abs(ayg) + DOMINANT) &&
                    (abs(axg) > abs(azg) + DOMINANT);

  bool ayDominant = (abs(ayg) > GTHRESHOLD) &&
                    (abs(ayg) > abs(axg) + DOMINANT) &&
                    (abs(ayg) > abs(azg) + DOMINANT);

  bool azDominant = (abs(azg) > GTHRESHOLD) &&
                    (abs(azg) > abs(axg) + DOMINANT) &&
                    (abs(azg) > abs(ayg) + DOMINANT);

  // Performing the according actions of the chosen dominant axis
  // Case where X axis is the dominant
  if (axDominant) {
  
    if (axg > 0) { 
      
      newPosture = "🤖 Robot lying on its back";
    }
    
    else {
      
      newPosture = "🤖 Robot lying on its front";
    }

    unsafePosture = true;
  }
  
  // Case where Y axis is the dominant
  else if (ayDominant) {
  
    if (ayg > 0) { 
      
      newPosture = "🤖 Robot lying on its right side";
    } 
    
    else {
      
      newPosture = "🤖 Robot lying on its left side";
    } 

    unsafePosture = true;
  }
  
  // Case where Z axis is the dominant
  else if (azDominant) {
  
    if (azg > 0) { 
      
      newPosture = "🟢 Robot upright";
    }

    else {
      
      newPosture = "🔄 Robot upside down";

      unsafePosture = true;
    }  
  }
  
  // Case where the robot is not on a definitive pose yet
  // like when is going to e.g fall on its left side 
  else {
  
    newPosture = "❓ Unclear posture yet.";
  }

  // Updating posture if it's stable for some ms
  // Ensuring the state after a period of time
  if (newPosture != lastPosture) {

      if (newPosture != pendingPosture) {
          
          pendingPosture = newPosture;
          
          lastPostureChangeTime = millis();
      }

      // Waiting for confirmation delay
      if (now - lastPostureChangeTime >= POSTURESTABILITYDELAY) {

          lastPosture = newPosture;
          
          blackBox("POSTURE: " + newPosture);

          // Allowing re-trigger for new unsafe posture
          postureSafeMode = false;
      }
  }
}


// This function is used to classify the
// terrain type where the robot is moving
void classifyTerrain() {
  
  // Data arrays with the latest accelerometer readings of X, Y, Z axis
  // The values are stored in circular buffers axData[], ayData[], azData[]
  axData[bufferIndex] = axg;
  ayData[bufferIndex] = ayg;
  azData[bufferIndex] = azg;

  
  // This makes bufferIndex go from 0 to TERRAIN_DATA_SIZE - 1
  // and then wrap around like a ring buffer 
  // keeping the most recent TERRAIN_DATA_SIZE samples
  bufferIndex = (bufferIndex + 1) % TERRAIN_DATA_SIZE;
  
  // Checking if there is free space
  // Case that the arrays are full, after one full cycle
  // Proceeding to classification after the buffer is full
  // when enough samples are collected
  if (bufferIndex == 0){
    
    bufferFilled = true;
  }
  
  // Case they aren't
  if (!bufferFilled) {
    
    return;
  }
  
  // Calculating Root Mean Square
  // Computing the average value of acceleration on each axis
  float axMean = 0, ayMean = 0, azMean = 0;
  
  for (int i = 0; i < TERRAIN_DATA_SIZE; i++) {
  
    axMean += axData[i];
    ayMean += ayData[i];
    azMean += azData[i];
  }
  
  axMean /= TERRAIN_DATA_SIZE;
  ayMean /= TERRAIN_DATA_SIZE;
  azMean /= TERRAIN_DATA_SIZE;

  // Calculating Standard deviation / Variance
  // Variance shows how much the acceleration values fluctuate
  float axVar = 0, ayVar = 0, azVar = 0;
  
  for (int i = 0; i < TERRAIN_DATA_SIZE; i++) {
  
    axVar += pow(axData[i] - axMean, 2);
    ayVar += pow(ayData[i] - ayMean, 2);
    azVar += pow(azData[i] - azMean, 2);
  }

  axVar /= TERRAIN_DATA_SIZE;
  ayVar /= TERRAIN_DATA_SIZE;
  azVar /= TERRAIN_DATA_SIZE;

  // Final variance gives a single value
  // that represents overall vibration in a 3D space
  float totalVariance = axVar + ayVar + azVar;

  // Checking the terrain type
  // based on g^2 units
  if (totalVariance < 0.02) {
  
    newTerrain = "🟩 Flat terrain";
  }
  
  else if (totalVariance < 0.65) {
  
    newTerrain = "🟨 Medium rough terrain";
  }
  
  else {
  
    newTerrain = "🟥 Rough terrain!";
  }

  // Ensuring the current state of the terrain type beforing changing
  // Preventing false triggers

  // Case where the type is about to change to a new one 
  if (newTerrain != currentTerrain) {
    
    if (newTerrain != pendingTerrain) {
    
      pendingTerrain = newTerrain;
    
      terrainChangeStartTime = millis();
    }

    // Counting current millis
    unsigned long current = millis();

    // Staying to the current terrain type for 2 seconds
    // to prevent false triger alerts
    if (current - terrainChangeStartTime >= terrainStabilityDelay) {
        
      currentTerrain = pendingTerrain;
      
      blackBox("✅ Terrain changed to: " + currentTerrain);
    }
  }
  
  // Case where the type is still the same
  else {
    
    // Resetting
    pendingTerrain = "";

    terrainChangeStartTime = 0;
  }
}
