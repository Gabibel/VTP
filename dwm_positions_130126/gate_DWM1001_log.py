import subprocess
import json
import csv
from datetime import datetime
import os
import argparse
import threading
import signal

# Argument parsing
parser = argparse.ArgumentParser(description="MQTT CSV Logger using mosquitto_sub")
parser.add_argument("--file", type=str, required=True, help="Name of the CSV file (without .csv)")
parser.add_argument("--dir", type=str, default=".", help="Directory to save the CSV file")
parser.add_argument("--time", type=int, default=30, help="Duration of logging in seconds")
args = parser.parse_args()

# Prepare CSV path
filename = args.file if args.file.endswith(".csv") else args.file + ".csv"
csv_filename = os.path.join(args.dir, filename)
os.makedirs(args.dir, exist_ok=True)

# Write header
with open(csv_filename, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["time", "nodeID", "positionX", "positionY", "positionZ", "quality"])

def process_message(topic, payload):
    if not topic.endswith("location"):
        return
    parts = topic.split('/')
    if len(parts) < 4:
        print("Unexpected topic format:", topic)
        return
    nodeID = parts[2]
    try:
        data = json.loads(payload)
    except Exception as e:
        print("Error decoding JSON:", e)
        return
    current_time = datetime.now().isoformat()
    position = data.get("position", {})
    row = [
        current_time,
        nodeID,
        position.get("x", ""),
        position.get("y", ""),
        position.get("z", ""),
        position.get("quality", "")
    ]
    try:
        with open(csv_filename, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
        print("Logged row:", row)
    except Exception as e:
        print("Error writing to CSV file:", e)

def main():
    print("Logging to:", os.path.abspath(csv_filename))
    print("Logging will run for", args.time, "seconds...")

    command = [
        "mosquitto_sub", "-h", "localhost", "-p", "1883",
        "-t", "#", "-u", "dwmuser", "-P", "dwmpass", "-v"
    ]

    # Launch mosquitto_sub with a new process group
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        preexec_fn=os.setsid
    )

    def stop_logging():
        print("⏰ Tempo encerrado. Encerrando grupo de processos...")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception as e:
            print("Erro ao encerrar processo:", e)

    timer = threading.Timer(args.time, stop_logging)
    timer.start()

    try:
        for line in process.stdout:
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                topic, payload = line.split(" ", 1)
                process_message(topic, payload)
            except ValueError:
                continue
    finally:
        timer.cancel()
        print("Logging finalizado.")

if __name__ == "__main__":
    main()

