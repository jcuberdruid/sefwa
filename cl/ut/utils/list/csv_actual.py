import os
import glob
import json
import sys
import csv

# If a specific run_note is passed as an argument, use it, else prompt the user
if len(sys.argv) > 1:
    target_run_note = sys.argv[1]
else:
    target_run_note = input("Please enter the target run_note: ")

perfPath = '../../../logs/tf_perf_logs/'

output_logs_array = []

# Loop through all directories in perfPath
for directory in os.listdir(perfPath):
    # Only consider items that are directories
    if os.path.isdir(os.path.join(perfPath, directory)):
        # Find all JSON files with the name format 'output_log*.json' within each directory
        pattern = os.path.join(perfPath, directory, 'output_log*.json')
        output_logs_array.extend(glob.glob(pattern))

filtered_logs = []

# For each found JSON file
for json_file in output_logs_array:
    with open(json_file, 'r') as file:
        log = json.load(file)
        # If the log's run_note matches the target and it has testing_subjects, add to filtered logs
        if log.get("run_note") == target_run_note and 'testing_subjects' in log and log['testing_subjects']:
            filtered_logs.append({
                'testing_subject': log['testing_subjects'][0],
                'accuracy': log['accuracy'],
                'run_note': log['run_note']
            })

# Write filtered logs to a CSV file
with open('filtered_logs.csv', 'w', newline='') as csvfile:
    fieldnames = ['testing_subject', 'accuracy', 'run_note']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for log in filtered_logs:
        writer.writerow(log)

print(f"Logs with run_note '{target_run_note}' have been written to 'filtered_logs.csv'.")

