import csv
import queue
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path

import serial
from serial.tools import list_ports


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

# Set this to your Arduino port.
#
# Run:
# python -m serial.tools.list_ports
#
# Or set PORT = None to automatically select a USB modem port.
PORT = "/dev/cu.usbmodem6076238102"

# Must match Serial.begin(...) in the Arduino sketch.
BAUD_RATE = 921600

# Number of CSV columns sent by the Arduino.
EXPECTED_ARDUINO_COLUMNS = 22

# Desired number of successfully completed trials.
DEFAULT_TARGET_REPETITIONS = 20
UNKNOWN_TARGET_REPETITIONS = 60


# ---------------------------------------------------------
# LABELS
# ---------------------------------------------------------

LABEL_NAMES = {
    0: "unknown",
    1: "start",
    2: "stop",
    3: "approve",
    4: "app",
    5: "meme",
    6: "generator",
    7: "please",
    8: "run",
    9: "localhost",
    10: "fast",
    11: "clench",
    12: "double_clench",
    13: "please_code_me_a",
    14: "how_is_the_weather",
}

LABEL_TYPES = {
    0: "unknown",
    1: "word",
    2: "word",
    3: "word",
    4: "word",
    5: "word",
    6: "word",
    7: "word",
    8: "word",
    9: "word",
    10: "word",
    11: "gesture",
    12: "gesture",
    13: "sentence",
    14: "sentence",
}

DISPLAY_NAMES = {
    0: "UNKNOWN / NON-TARGET",
    1: "START",
    2: "STOP",
    3: "APPROVE",
    4: "APP",
    5: "MEME",
    6: "GENERATOR",
    7: "PLEASE",
    8: "RUN",
    9: "LOCALHOST",
    10: "FAST",
    11: "CLENCH",
    12: "DOUBLE CLENCH",
    13: "PLEASE CODE ME A",
    14: "HOW IS THE WEATHER",
}

# These commands must match the updated Arduino sketch.
COMMAND_TO_LABEL = {
    "u": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "0": 10,
    "c": 11,
    "d": 12,
    "p": 13,
    "w": 14,
}

LABEL_TO_COMMAND = {
    label: command
    for command, label in COMMAND_TO_LABEL.items()
}

# Allows full names to be entered instead of shortcuts.
COMMAND_ALIASES = {
    "unknown": "u",
    "start": "1",
    "stop": "2",
    "approve": "3",
    "app": "4",
    "meme": "5",
    "generator": "6",
    "please": "7",
    "run": "8",
    "localhost": "9",
    "fast": "0",
    "clench": "c",
    "double": "d",
    "double clench": "d",
    "double_clench": "d",
    "please code me a": "p",
    "please_code_me_a": "p",
    "how is the weather": "w",
    "how_is_the_weather": "w",
}

TARGET_REPETITIONS = {
    label: (
        UNKNOWN_TARGET_REPETITIONS
        if label == 0
        else DEFAULT_TARGET_REPETITIONS
    )
    for label in LABEL_NAMES
}


# ---------------------------------------------------------
# CSV FORMAT
# ---------------------------------------------------------

CSV_HEADER = [
    "time_us",
    "dt_us",
    "session_id",
    "trial_id",
    "sample_index",
    "phase",
    "jaw_ax_mg",
    "jaw_ay_mg",
    "jaw_az_mg",
    "jaw_gx_cdeg_s",
    "jaw_gy_cdeg_s",
    "jaw_gz_cdeg_s",
    "ref_ax_mg",
    "ref_ay_mg",
    "ref_az_mg",
    "ref_gx_cdeg_s",
    "ref_gy_cdeg_s",
    "ref_gz_cdeg_s",
    "label",
    "ready_mask",
    "late_us",
    "read_span_us",
    "target_name",
    "target_type",
]


# ---------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------

command_queue: queue.Queue[str] = queue.Queue()
stop_event = threading.Event()
print_lock = threading.Lock()

trial_counts: Counter[int] = Counter()

current_label = 0
current_target = LABEL_NAMES[0]
current_trial_active = False


# ---------------------------------------------------------
# TERMINAL HELPERS
# ---------------------------------------------------------

def safe_print(*values: object, **kwargs: object) -> None:
    with print_lock:
        print(*values, **kwargs)


def print_commands() -> None:
    safe_print()
    safe_print("=" * 68)
    safe_print("RECORDING COMMANDS")
    safe_print("=" * 68)
    safe_print("1 = START                 2 = STOP")
    safe_print("3 = APPROVE               4 = APP")
    safe_print("5 = MEME                  6 = GENERATOR")
    safe_print("7 = PLEASE                8 = RUN")
    safe_print("9 = LOCALHOST             0 = FAST")
    safe_print("c = CLENCH                d = DOUBLE CLENCH")
    safe_print("p = PLEASE CODE ME A      w = HOW IS THE WEATHER")
    safe_print("u = UNKNOWN / NON-TARGET")
    safe_print()
    safe_print("x = abort current trial")
    safe_print("s = show collection progress")
    safe_print("q = finish recording and save")
    safe_print("=" * 68)
    safe_print()


def print_progress() -> None:
    safe_print()
    safe_print("=" * 68)
    safe_print("COLLECTION PROGRESS")
    safe_print("=" * 68)

    for label in sorted(LABEL_NAMES):
        completed = trial_counts[label]
        target = TARGET_REPETITIONS[label]
        display_name = DISPLAY_NAMES[label]

        safe_print(
            f"{label:>2}  "
            f"{display_name:<24} "
            f"{completed:>3}/{target:<3}"
        )

    safe_print("=" * 68)
    safe_print()


def normalize_user_command(user_input: str) -> str | None:
    normalized = " ".join(
        user_input.strip().lower().replace("-", " ").split()
    )

    if normalized in COMMAND_TO_LABEL:
        return normalized

    return COMMAND_ALIASES.get(normalized)


# ---------------------------------------------------------
# TERMINAL INPUT THREAD
# ---------------------------------------------------------

def keyboard_worker() -> None:
    print_commands()

    while not stop_event.is_set():
        try:
            user_input = input("Enter command: ")
        except EOFError:
            user_input = "q"

        normalized = user_input.strip().lower()

        if normalized == "q":
            stop_event.set()
            return

        if normalized == "x":
            command_queue.put("x")
            continue

        if normalized == "s":
            print_progress()
            continue

        command = normalize_user_command(user_input)

        if command is not None:
            command_queue.put(command)
        elif normalized:
            safe_print(
                "Invalid command. Enter a shortcut, target name, "
                "s, x, or q."
            )


# ---------------------------------------------------------
# SERIAL EVENT PARSING
# ---------------------------------------------------------

def parse_event_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}

    for section in line.split(",")[1:]:
        if "=" not in section:
            continue

        key, value = section.split("=", 1)
        fields[key.strip()] = value.strip()

    return fields


def set_current_target_from_event(line: str) -> None:
    global current_label
    global current_target

    fields = parse_event_fields(line)

    try:
        event_label = int(fields.get("label", ""))
    except ValueError:
        event_label = -1

    if event_label in LABEL_NAMES:
        current_label = event_label
        current_target = LABEL_NAMES[event_label]
        return

    # Fall back to the word/target field if supplied.
    event_target = fields.get(
        "target",
        fields.get("word", "unknown"),
    )

    current_target = event_target

    for label, name in LABEL_NAMES.items():
        if name == event_target:
            current_label = label
            return

    current_label = 0


# ---------------------------------------------------------
# USER INSTRUCTIONS
# ---------------------------------------------------------

def print_action_instruction(label: int) -> None:
    display_name = DISPLAY_NAMES.get(label, "UNKNOWN")

    if label == 11:
        safe_print("CLENCH ONCE NOW")
        safe_print("Keep your head still and perform one normal clench.")

    elif label == 12:
        safe_print("DOUBLE CLENCH NOW")
        safe_print("Clench twice quickly, then stop moving.")

    elif label in {13, 14}:
        safe_print(f"SILENTLY MOUTH: “{display_name}”")
        safe_print("Mouth the complete phrase once, naturally, with no sound.")

    elif label == 0:
        safe_print("PERFORM ONE UNKNOWN / NON-TARGET ACTION")
        safe_print(
            "Examples: swallow, random word, head turn, smile, "
            "chew, adjust the sensor, or remain still."
        )

    else:
        safe_print(f"SILENTLY MOUTH: “{display_name}”")
        safe_print("Mouth the word once, naturally, with no sound.")


def print_event_message(line: str) -> None:
    global current_trial_active

    if line.startswith("# TRIAL_START"):
        set_current_target_from_event(line)
        current_trial_active = True

        display_name = DISPLAY_NAMES.get(
            current_label,
            current_target.upper(),
        )

        completed = trial_counts[current_label]
        target = TARGET_REPETITIONS.get(
            current_label,
            DEFAULT_TARGET_REPETITIONS,
        )

        safe_print()
        safe_print("=" * 68)
        safe_print(f"NEW TRIAL: {display_name}")
        safe_print(f"Progress before trial: {completed}/{target}")
        safe_print("GET READY — keep your head and jaw completely still.")
        safe_print("=" * 68)

    elif line.startswith("# MOUTH_NOW"):
        safe_print()
        safe_print("█" * 68)
        print_action_instruction(current_label)
        safe_print("█" * 68)

    elif line.startswith("# REST_NOW"):
        safe_print()
        safe_print("-" * 68)
        safe_print("STOP — relax your jaw and remain completely still.")
        safe_print("-" * 68)

    elif line.startswith("# TRIAL_DONE"):
        current_trial_active = False
        trial_counts[current_label] += 1

        completed = trial_counts[current_label]
        target = TARGET_REPETITIONS.get(
            current_label,
            DEFAULT_TARGET_REPETITIONS,
        )

        safe_print()
        safe_print("=" * 68)
        safe_print(
            f"TRIAL COMPLETE: "
            f"{DISPLAY_NAMES.get(current_label, current_target.upper())}"
        )
        safe_print(f"Progress: {completed}/{target}")

        if completed >= target:
            safe_print("TARGET REPETITION COUNT REACHED FOR THIS CLASS.")

        safe_print("Enter the next command.")
        safe_print("=" * 68)
        safe_print()

    elif line.startswith("# TRIAL_ABORTED"):
        current_trial_active = False

        safe_print()
        safe_print("=" * 68)
        safe_print("TRIAL ABORTED — this trial was not counted.")
        safe_print("=" * 68)
        safe_print()

    elif line.startswith("# BUSY"):
        safe_print()
        safe_print("A trial is already running. Wait for it to finish.")
        safe_print()

    elif line.startswith("# ERROR"):
        safe_print()
        safe_print(line)
        safe_print()

    elif line.startswith("# Both IMUs found"):
        safe_print()
        safe_print("Both IMUs connected successfully.")
        safe_print()

    else:
        safe_print(line)


# ---------------------------------------------------------
# PORT SELECTION
# ---------------------------------------------------------

def find_serial_port() -> str:
    if PORT:
        return PORT

    available_ports = list(list_ports.comports())

    for port in available_ports:
        device = port.device.lower()

        if (
            "usbmodem" in device
            or "usbserial" in device
            or "arduino" in port.description.lower()
        ):
            return port.device

    devices = ", ".join(port.device for port in available_ports)

    raise RuntimeError(
        "No likely Arduino serial port was found. "
        f"Available ports: {devices or 'none'}"
    )


# ---------------------------------------------------------
# MAIN RECORDER
# ---------------------------------------------------------

def main() -> None:
    output_folder = Path("imu_data")
    output_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    csv_path = output_folder / f"silent_speech_{timestamp}.csv"
    event_path = output_folder / f"silent_speech_{timestamp}_events.txt"
    summary_path = output_folder / f"silent_speech_{timestamp}_summary.csv"

    rows_written = 0
    malformed_rows = 0

    try:
        serial_port = find_serial_port()
    except RuntimeError as error:
        safe_print(f"PORT ERROR: {error}")
        return

    safe_print(f"Opening serial port: {serial_port}")
    safe_print(f"Baud rate: {BAUD_RATE}")
    safe_print(f"CSV will be saved to: {csv_path}")
    safe_print()

    keyboard_thread = threading.Thread(
        target=keyboard_worker,
        daemon=True,
    )
    keyboard_thread.start()

    try:
        with (
            serial.Serial(
                port=serial_port,
                baudrate=BAUD_RATE,
                timeout=0.2,
            ) as arduino,
            csv_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as csv_file,
            event_path.open(
                "w",
                encoding="utf-8",
            ) as event_file,
        ):
            writer = csv.writer(csv_file)
            writer.writerow(CSV_HEADER)
            csv_file.flush()

            # Remove old serial bytes left in the input buffer.
            arduino.reset_input_buffer()

            while not stop_event.is_set():
                while not command_queue.empty():
                    command = command_queue.get_nowait()

                    arduino.write(command.encode("ascii"))
                    arduino.flush()

                raw_line = arduino.readline()

                if not raw_line:
                    continue

                line = raw_line.decode(
                    "utf-8",
                    errors="replace",
                ).strip()

                if not line:
                    continue

                # Arduino event/status lines.
                if line.startswith("#"):
                    event_file.write(line + "\n")
                    event_file.flush()

                    print_event_message(line)
                    continue

                # Ignore the Arduino CSV header.
                if line.startswith("time_us,"):
                    continue

                values = line.split(",")

                if len(values) != EXPECTED_ARDUINO_COLUMNS:
                    malformed_rows += 1
                    safe_print(
                        f"Skipped malformed row "
                        f"({len(values)} columns)."
                    )
                    continue

                try:
                    int(values[0])    # time_us
                    int(values[1])    # dt_us
                    int(values[2])    # session_id
                    int(values[3])    # trial_id
                    int(values[4])    # sample_index
                    int(values[5])    # phase

                    label = int(values[18])
                    ready_mask = int(values[19])

                except ValueError:
                    malformed_rows += 1
                    safe_print("Skipped a row containing invalid numbers.")
                    continue

                if label not in LABEL_NAMES:
                    malformed_rows += 1
                    safe_print(f"Skipped row with unknown label: {label}")
                    continue

                target_name = LABEL_NAMES[label]
                target_type = LABEL_TYPES[label]

                writer.writerow(
                    values + [
                        target_name,
                        target_type,
                    ]
                )

                rows_written += 1

                if ready_mask != 15 and rows_written % 100 == 0:
                    safe_print(
                        "Warning: some rows do not have ready_mask=15."
                    )

                # Regular flushing reduces data loss if interrupted.
                if rows_written % 25 == 0:
                    csv_file.flush()

            csv_file.flush()
            event_file.flush()

    except serial.SerialException as error:
        safe_print()
        safe_print(f"SERIAL ERROR: {error}")
        safe_print(
            "Check the PORT setting and close Arduino Serial Monitor."
        )

    except KeyboardInterrupt:
        safe_print()
        safe_print("Recording stopped with Ctrl+C.")

    finally:
        stop_event.set()

        with summary_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as summary_file:
            summary_writer = csv.writer(summary_file)

            summary_writer.writerow(
                [
                    "label",
                    "target_name",
                    "target_type",
                    "completed_trials",
                    "target_trials",
                ]
            )

            for label in sorted(LABEL_NAMES):
                summary_writer.writerow(
                    [
                        label,
                        LABEL_NAMES[label],
                        LABEL_TYPES[label],
                        trial_counts[label],
                        TARGET_REPETITIONS[label],
                    ]
                )

        safe_print()
        safe_print("=" * 68)
        safe_print("RECORDING FINISHED")
        safe_print("=" * 68)
        safe_print(f"Sensor rows saved: {rows_written}")
        safe_print(f"Malformed rows skipped: {malformed_rows}")
        safe_print(f"CSV file: {csv_path}")
        safe_print(f"Event log: {event_path}")
        safe_print(f"Summary file: {summary_path}")
        safe_print("=" * 68)

        print_progress()


if __name__ == "__main__":
    main()