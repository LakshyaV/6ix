import pandas as pd

INPUT_FILE = "/Users/laky.vasu/Desktop/6ix/imu_data/silent_speech_2026-07-18_09-32-43.csv"
OUTPUT_FILE = "/imu_data_cleaned.csv"

df = pd.read_csv(INPUT_FILE)

# Find the first recorded STOP sample
stop_rows = df[df["target_name"].astype(str).str.strip().str.lower() == "stop"]

if stop_rows.empty:
    raise ValueError("No STOP trials were found.")

first_stop = stop_rows.iloc[0]
session_id = first_stop["session_id"]
trial_id = first_stop["trial_id"]

# Remove every sensor row belonging to that trial
remove_mask = (
    (df["session_id"] == session_id)
    & (df["trial_id"] == trial_id)
)

removed_rows = int(remove_mask.sum())
cleaned_df = df.loc[~remove_mask].copy()
cleaned_df.to_csv(OUTPUT_FILE, index=False)

print(f"Removed first STOP trial:")
print(f"  Session: {session_id}")
print(f"  Trial:   {trial_id}")
print(f"  Rows:    {removed_rows}")
print(f"Saved cleaned data to {OUTPUT_FILE}")