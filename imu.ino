#include <Wire.h>
#include <Arduino_LSM6DSOX.h>
#include <math.h>

// IMU 1: jaw/TMJ sensor on Qwiic port
// IMU 2: reference sensor behind the ear
LSM6DSOXClass jawIMU(Wire1, 0x6A);
LSM6DSOXClass refIMU(Wire, 0x6A);

// Increase this whenever you remove/remount the sensors.
constexpr uint16_t SESSION_ID = 1;

// 100 Hz sampling
constexpr uint32_t SAMPLE_PERIOD_US = 10000;

// Rest periods
constexpr uint32_t PRE_REST_US = 1000000;
constexpr uint32_t POST_REST_US = 1000000;

// Words and gestures get 1.5 seconds.
// Sentences get 3 seconds.
constexpr uint32_t NORMAL_ACTION_US = 1500000;
constexpr uint32_t SENTENCE_ACTION_US = 3000000;


// ---------------------------------------------------------
// LABELS
// ---------------------------------------------------------

enum TargetLabel : uint8_t {
  LABEL_UNKNOWN = 0,
  LABEL_START = 1,
  LABEL_STOP = 2,
  LABEL_APPROVE = 3,
  LABEL_APP = 4,
  LABEL_MEME = 5,
  LABEL_GENERATOR = 6,
  LABEL_PLEASE = 7,
  LABEL_RUN = 8,
  LABEL_LOCALHOST = 9,
  LABEL_FAST = 10,
  LABEL_CLENCH = 11,
  LABEL_DOUBLE_CLENCH = 12,
  LABEL_PLEASE_CODE_ME_A = 13,
  LABEL_HOW_IS_THE_WEATHER = 14
};

enum TrialPhase : uint8_t {
  PHASE_PRE_REST = 0,
  PHASE_ACTION = 1,
  PHASE_POST_REST = 2
};


// ---------------------------------------------------------
// TRIAL STATE
// ---------------------------------------------------------

bool trialActive = false;

uint8_t activeLabel = LABEL_UNKNOWN;
uint8_t activePhase = PHASE_PRE_REST;

uint32_t activeActionDurationUs = NORMAL_ACTION_US;

uint32_t trialId = 0;
uint32_t trialStartUs = 0;
uint32_t nextSampleUs = 0;
uint32_t previousSampleUs = 0;

uint16_t sampleIndex = 0;

uint32_t missedDeadlines = 0;
uint32_t readErrors = 0;


// ---------------------------------------------------------
// LABEL HELPERS
// ---------------------------------------------------------

const char* labelName(uint8_t label) {
  switch (label) {
    case LABEL_START:
      return "start";

    case LABEL_STOP:
      return "stop";

    case LABEL_APPROVE:
      return "approve";

    case LABEL_APP:
      return "app";

    case LABEL_MEME:
      return "meme";

    case LABEL_GENERATOR:
      return "generator";

    case LABEL_PLEASE:
      return "please";

    case LABEL_RUN:
      return "run";

    case LABEL_LOCALHOST:
      return "localhost";

    case LABEL_FAST:
      return "fast";

    case LABEL_CLENCH:
      return "clench";

    case LABEL_DOUBLE_CLENCH:
      return "double_clench";

    case LABEL_PLEASE_CODE_ME_A:
      return "please_code_me_a";

    case LABEL_HOW_IS_THE_WEATHER:
      return "how_is_the_weather";

    default:
      return "unknown";
  }
}

uint32_t actionDurationForLabel(uint8_t label) {
  if (
    label == LABEL_PLEASE_CODE_ME_A ||
    label == LABEL_HOW_IS_THE_WEATHER
  ) {
    return SENTENCE_ACTION_US;
  }

  return NORMAL_ACTION_US;
}

uint32_t totalTrialDurationUs() {
  return PRE_REST_US + activeActionDurationUs + POST_REST_US;
}


// ---------------------------------------------------------
// PHASE CONTROL
// ---------------------------------------------------------

uint8_t determinePhase(uint32_t elapsedUs) {
  if (elapsedUs < PRE_REST_US) {
    return PHASE_PRE_REST;
  }

  if (elapsedUs < PRE_REST_US + activeActionDurationUs) {
    return PHASE_ACTION;
  }

  return PHASE_POST_REST;
}

void updatePhase(uint8_t newPhase) {
  if (newPhase == activePhase) {
    return;
  }

  activePhase = newPhase;

  if (activePhase == PHASE_ACTION) {
    Serial.print("# MOUTH_NOW,trial=");
    Serial.println(trialId);
  }

  else if (activePhase == PHASE_POST_REST) {
    Serial.print("# REST_NOW,trial=");
    Serial.println(trialId);
  }
}


// ---------------------------------------------------------
// TRIAL CONTROL
// ---------------------------------------------------------

void startTrial(uint8_t label) {
  if (trialActive) {
    Serial.println("# BUSY: finish or abort the current trial.");
    return;
  }

  trialId++;

  activeLabel = label;
  activePhase = PHASE_PRE_REST;
  activeActionDurationUs = actionDurationForLabel(label);

  sampleIndex = 0;
  missedDeadlines = 0;
  readErrors = 0;
  previousSampleUs = 0;

  trialStartUs = micros();
  nextSampleUs = trialStartUs;

  trialActive = true;

  Serial.print("# TRIAL_START,session=");
  Serial.print(SESSION_ID);

  Serial.print(",trial=");
  Serial.print(trialId);

  Serial.print(",label=");
  Serial.print(activeLabel);

  Serial.print(",word=");
  Serial.println(labelName(activeLabel));
}

void endTrial(bool aborted) {
  trialActive = false;

  if (aborted) {
    Serial.print("# TRIAL_ABORTED");
  } else {
    Serial.print("# TRIAL_DONE");
  }

  Serial.print(",session=");
  Serial.print(SESSION_ID);

  Serial.print(",trial=");
  Serial.print(trialId);

  Serial.print(",samples=");
  Serial.print(sampleIndex);

  Serial.print(",missed_deadlines=");
  Serial.print(missedDeadlines);

  Serial.print(",read_errors=");
  Serial.println(readErrors);
}


// ---------------------------------------------------------
// COMMANDS FROM PYTHON
// ---------------------------------------------------------

void readCommands() {
  while (Serial.available() > 0) {
    char command = Serial.read();

    switch (command) {
      case 'u':
      case 'U':
        startTrial(LABEL_UNKNOWN);
        break;

      case '1':
        startTrial(LABEL_START);
        break;

      case '2':
        startTrial(LABEL_STOP);
        break;

      case '3':
        startTrial(LABEL_APPROVE);
        break;

      case '4':
        startTrial(LABEL_APP);
        break;

      case '5':
        startTrial(LABEL_MEME);
        break;

      case '6':
        startTrial(LABEL_GENERATOR);
        break;

      case '7':
        startTrial(LABEL_PLEASE);
        break;

      case '8':
        startTrial(LABEL_RUN);
        break;

      case '9':
        startTrial(LABEL_LOCALHOST);
        break;

      case '0':
        startTrial(LABEL_FAST);
        break;

      case 'c':
      case 'C':
        startTrial(LABEL_CLENCH);
        break;

      case 'd':
      case 'D':
        startTrial(LABEL_DOUBLE_CLENCH);
        break;

      case 'p':
      case 'P':
        startTrial(LABEL_PLEASE_CODE_ME_A);
        break;

      case 'w':
      case 'W':
        startTrial(LABEL_HOW_IS_THE_WEATHER);
        break;

      case 'x':
      case 'X':
        if (trialActive) {
          endTrial(true);
        }
        break;

      default:
        break;
    }
  }
}


// ---------------------------------------------------------
// SETUP
// ---------------------------------------------------------

void setup() {
  Serial.begin(921600);
  delay(2000);

  Wire.begin();
  Wire1.begin();

  if (!jawIMU.begin() || !refIMU.begin()) {
    Serial.println("# ERROR: One or both IMUs were not found.");

    while (true) {
      delay(1000);
    }
  }

  Serial.println("# Both IMUs found.");

  Serial.print("# jaw_accel_hz=");
  Serial.println(jawIMU.accelerationSampleRate());

  Serial.print("# jaw_gyro_hz=");
  Serial.println(jawIMU.gyroscopeSampleRate());

  Serial.print("# ref_accel_hz=");
  Serial.println(refIMU.accelerationSampleRate());

  Serial.print("# ref_gyro_hz=");
  Serial.println(refIMU.gyroscopeSampleRate());

  Serial.println(
    "# Commands: "
    "1=start, 2=stop, 3=approve, 4=app, "
    "5=meme, 6=generator, 7=please, 8=run, "
    "9=localhost, 0=fast, c=clench, "
    "d=double_clench, p=please_code_me_a, "
    "w=how_is_the_weather, u=unknown, x=abort"
  );

  // This header has exactly 22 columns.
  Serial.println(
    "time_us,dt_us,session_id,trial_id,sample_index,phase,"
    "jaw_ax_mg,jaw_ay_mg,jaw_az_mg,"
    "jaw_gx_cdeg_s,jaw_gy_cdeg_s,jaw_gz_cdeg_s,"
    "ref_ax_mg,ref_ay_mg,ref_az_mg,"
    "ref_gx_cdeg_s,ref_gy_cdeg_s,ref_gz_cdeg_s,"
    "label,ready_mask,late_us,read_span_us"
  );
}


// ---------------------------------------------------------
// MAIN LOOP
// ---------------------------------------------------------

void loop() {
  readCommands();

  if (!trialActive) {
    return;
  }

  uint32_t now = micros();
  uint32_t elapsedUs = now - trialStartUs;

  if (elapsedUs >= totalTrialDurationUs()) {
    endTrial(false);
    return;
  }

  updatePhase(determinePhase(elapsedUs));

  // Wait until the next 100 Hz sample time.
  if ((int32_t)(now - nextSampleUs) < 0) {
    return;
  }

  int32_t lateUs = (int32_t)(now - nextSampleUs);

  if (lateUs >= (int32_t)SAMPLE_PERIOD_US) {
    missedDeadlines +=
      (uint32_t)lateUs / SAMPLE_PERIOD_US;

    nextSampleUs = now + SAMPLE_PERIOD_US;
  } else {
    nextSampleUs += SAMPLE_PERIOD_US;
  }

  uint8_t readyMask = 0;

  if (jawIMU.accelerationAvailable()) {
    readyMask |= 1 << 0;
  }

  if (jawIMU.gyroscopeAvailable()) {
    readyMask |= 1 << 1;
  }

  if (refIMU.accelerationAvailable()) {
    readyMask |= 1 << 2;
  }

  if (refIMU.gyroscopeAvailable()) {
    readyMask |= 1 << 3;
  }

  float jawAx, jawAy, jawAz;
  float jawGx, jawGy, jawGz;

  float refAx, refAy, refAz;
  float refGx, refGy, refGz;

  uint32_t readStartUs = micros();

  bool readOkay =
    jawIMU.readAcceleration(jawAx, jawAy, jawAz) &&
    jawIMU.readGyroscope(jawGx, jawGy, jawGz) &&
    refIMU.readAcceleration(refAx, refAy, refAz) &&
    refIMU.readGyroscope(refGx, refGy, refGz);

  uint32_t readEndUs = micros();

  if (!readOkay) {
    readErrors++;
    return;
  }

  uint32_t sampleTimeUs =
    readStartUs + ((readEndUs - readStartUs) / 2);

  uint32_t dtUs =
    previousSampleUs == 0
      ? 0
      : sampleTimeUs - previousSampleUs;

  previousSampleUs = sampleTimeUs;

  int jawAxMg = (int)lroundf(jawAx * 1000.0f);
  int jawAyMg = (int)lroundf(jawAy * 1000.0f);
  int jawAzMg = (int)lroundf(jawAz * 1000.0f);

  int32_t jawGxCdeg = (int32_t)lroundf(jawGx * 100.0f);
  int32_t jawGyCdeg = (int32_t)lroundf(jawGy * 100.0f);
  int32_t jawGzCdeg = (int32_t)lroundf(jawGz * 100.0f);

  int refAxMg = (int)lroundf(refAx * 1000.0f);
  int refAyMg = (int)lroundf(refAy * 1000.0f);
  int refAzMg = (int)lroundf(refAz * 1000.0f);

  int32_t refGxCdeg = (int32_t)lroundf(refGx * 100.0f);
  int32_t refGyCdeg = (int32_t)lroundf(refGy * 100.0f);
  int32_t refGzCdeg = (int32_t)lroundf(refGz * 100.0f);

  char row[256];

  int rowLength = snprintf(
    row,
    sizeof(row),

    "%lu,%lu,%u,%lu,%u,%u,"
    "%d,%d,%d,%ld,%ld,%ld,"
    "%d,%d,%d,%ld,%ld,%ld,"
    "%u,%u,%ld,%lu\n",

    (unsigned long)sampleTimeUs,
    (unsigned long)dtUs,
    SESSION_ID,
    (unsigned long)trialId,
    sampleIndex,
    activePhase,

    jawAxMg,
    jawAyMg,
    jawAzMg,
    (long)jawGxCdeg,
    (long)jawGyCdeg,
    (long)jawGzCdeg,

    refAxMg,
    refAyMg,
    refAzMg,
    (long)refGxCdeg,
    (long)refGyCdeg,
    (long)refGzCdeg,

    activeLabel,
    readyMask,
    (long)lateUs,
    (unsigned long)(readEndUs - readStartUs)
  );

  if (rowLength > 0 && rowLength < (int)sizeof(row)) {
    Serial.write(
      reinterpret_cast<const uint8_t*>(row),
      rowLength
    );
  }

  sampleIndex++;
}