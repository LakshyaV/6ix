#include <Servo.h>

Servo fanServo;

const int SERVO_PIN = 9;

// Avoid forcing the servo against its physical limits.
const int MIN_ANGLE = 20;
const int MAX_ANGLE = 160;
const int STOP_ANGLE = 90;

// Lower value = faster movement.
const unsigned long STEP_INTERVAL_MS = 1;

bool running = false;
int currentAngle = STOP_ANGLE;
int direction = 1;
unsigned long lastStepTime = 0;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(50);

  fanServo.attach(SERVO_PIN);
  fanServo.write(STOP_ANGLE);

  Serial.println("READY");
}

void loop() {
  // Check for START or STOP commands.
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    command.toUpperCase();

    if (command == "START") {
      running = true;
      Serial.println("SERVO_STARTED");
    }

    else if (command == "STOP") {
      running = false;
      currentAngle = STOP_ANGLE;
      fanServo.write(STOP_ANGLE);
      Serial.println("SERVO_STOPPED");
    }
  }

  // Continuously sweep without blocking serial commands.
  if (running && millis() - lastStepTime >= STEP_INTERVAL_MS) {
    lastStepTime = millis();

    currentAngle += direction;

    if (currentAngle >= MAX_ANGLE) {
      currentAngle = MAX_ANGLE;
      direction = -1;
    }

    else if (currentAngle <= MIN_ANGLE) {
      currentAngle = MIN_ANGLE;
      direction = 1;
    }

    fanServo.write(currentAngle);
  }
}