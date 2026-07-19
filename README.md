# 6ix Jaw Gestures — Arduino UNO Q App Lab

This app moves the working live jaw-gesture pipeline from the Mac onto the UNO Q:

- **STM32 MCU:** samples the external LSM6DSOX at `0x6A` as the jaw IMU and the built-in IMU as the reference.
- **Qualcomm Linux:** runs baseline learning, event segmentation, feature extraction, wake detection, and command classification.
- **Bridge:** transfers buffered IMU samples internally; USB serial and `pyserial` are not used.

## Files that must be present

```text
6ix_jaw_gesture_app_lab/
├── app.yaml
├── README.md
├── python/
│   ├── main.py
│   ├── jaw_gesture_ml.py
│   ├── requirements.txt
│   └── models/
│       ├── wake_model.joblib
│       └── command_model.joblib
└── sketch/
    ├── sketch.ino
    └── sketch.yaml
```

## One thing this ZIP cannot include

The two trained `.joblib` files were created on your Mac and were not attached to ChatGPT. Copy them from your project:

```text
artifacts/jaw_gesture/session_real_002/wake_model.joblib
artifacts/jaw_gesture/session_real_002/command_model.joblib
```

into:

```text
python/models/
```

## Transfer into App Lab

1. Extract this ZIP.
2. Open the extracted folder as an app in Arduino App Lab, or copy its files over the blank `6ix` app using the same folder structure.
3. Add the two `.joblib` files to `python/models/`.
4. Click **Run**.
5. The first run may take several minutes because App Lab installs NumPy, scikit-learn, and joblib.

## Expected console output

```text
Starting 6ix jaw-gesture app on Qualcomm Linux...
Loaded models: ...
Wake-label inversion: ON
STM32 bridge ready: sensors_ready=1,jaw=external_0x6A,reference=builtin,...
Learning neutral baseline... 15/30
Learning neutral baseline... 30/30
```

Then use the same flow as before:

1. Keep your jaw neutral while the baseline is learned.
2. Clench once.
3. Return to neutral.
4. Perform one command gesture.

Command mapping:

```text
shift_right   -> START
shift_left    -> STOP
push_forward  -> MAKE_APP
pull_backward -> WEATHER
```

## Important settings

The wake-label inversion is enabled by default because that is the configuration that made the Mac live test work. To disable it, set the container environment variable `INVERT_WAKE_LABELS=0` or change the constant in `python/main.py`.

Enable motion-score logs by setting `JAW_DEBUG=1` or changing `DEBUG` in `python/main.py`.

## Model-version issue

If model loading reports a scikit-learn version incompatibility, run this inside the working Mac `6ix` environment:

```bash
python -c "import sklearn; print(sklearn.__version__)"
```

Then pin that version in `python/requirements.txt`, for example:

```text
scikit-learn==1.7.2
```

## No Mac command is required

Do **not** run `python live_jaw_gesture_test.py` on the Mac for this version. App Lab starts `python/main.py` automatically on Qualcomm Linux.
