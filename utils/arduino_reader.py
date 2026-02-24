"""
Arduino Sensor Reader - FINAL VERSION
Reads real sensor data from Arduino via Serial
"""

import serial
import threading
import time
from datetime import datetime


class ArduinoReader:
    def __init__(self, port='COM7', baud_rate=9600):
        self.port = port
        self.baud_rate = baud_rate
        self.serial_connection = None
        self.is_connected = False
        self.latest_data = None
        self.running = False

        self.default_data = {
            'heart_rate': 0,
            'body_temp': 0.0,
            'spo2': 0,
            'activity_level': 0.0,
            'ambient_temp': 0.0,
            'humidity': 0.0,
            'timestamp': datetime.now().isoformat(),
            'source': 'arduino',
            'status': 'connecting',
            'sheep_id': 'sheep_001',
            'hr_status': 'no_finger'  # ✨ HR status
        }

    def connect(self):
        """Connect ONCE at startup"""
        if self.is_connected:
            return True

        try:
            print(f"Connecting to Arduino on {self.port}...")
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=2
            )
            self.is_connected = True
            print(f"✅ Connected to Arduino on {self.port}")
            time.sleep(2)

            self.running = True
            self.reading_thread = threading.Thread(
                target=self._read_loop,
                daemon=True
            )
            self.reading_thread.start()
            print("📡 Started reading Arduino data...")
            return True

        except serial.SerialException as e:
            print(f"❌ Arduino connection failed: {e}")
            self.is_connected = False
            return False

    def _read_loop(self):
        """Background thread reads Arduino continuously"""
        while self.running:
            try:
                if self.serial_connection and \
                   self.serial_connection.is_open:
                    line = self.serial_connection.readline()
                    line = line.decode('utf-8').strip()
                    if line and 'HR:' in line and 'TEMP:' in line:
                        parsed = self._parse_data(line)
                        if parsed:
                            self.latest_data = parsed
            except Exception:
                time.sleep(0.1)
                continue

    def _parse_data(self, raw_data):
        """
        Parse Arduino data string
        Format: HR:xx,TEMP:xx,SPO2:xx,ACT:xx,AMB:xx,HUM:xx,HRSTATUS:xx
        """
        try:
            data = {}
            parts = raw_data.split(',')

            for part in parts:
                if ':' in part:
                    key, value = part.split(':', 1)
                    key   = key.strip()
                    value = value.strip()

                    if key == 'HR':
                        data['heart_rate'] = int(float(value))
                    elif key == 'TEMP':
                        data['body_temp'] = float(value)
                    elif key == 'SPO2':
                        data['spo2'] = int(float(value))
                    elif key == 'ACT':
                        data['activity_level'] = float(value)
                    elif key == 'AMB':
                        data['ambient_temp'] = float(value)
                    elif key == 'HUM':
                        data['humidity'] = float(value)
                    elif key == 'HRSTATUS':
                        data['hr_status'] = value  # ✨ Parse HR status

            # Validate required fields
            required = [
                'heart_rate', 'body_temp', 'spo2',
                'activity_level', 'ambient_temp', 'humidity'
            ]

            if all(k in data for k in required):
                data['timestamp'] = datetime.now().isoformat()
                data['source']    = 'arduino'
                data['status']    = 'connected'
                data['sheep_id']  = 'sheep_001'

                # Default hr_status if not received
                if 'hr_status' not in data:
                    data['hr_status'] = 'no_finger'

                return data

            return None

        except Exception:
            return None

    def get_reading(self):
        """Return latest data - NEVER reconnects!"""
        if self.latest_data:
            return self.latest_data
        self.default_data['timestamp'] = datetime.now().isoformat()
        return self.default_data

    def disconnect(self):
        """Disconnect from Arduino"""
        self.running = False
        if self.serial_connection and \
           self.serial_connection.is_open:
            self.serial_connection.close()
        self.is_connected = False
        print("Disconnected from Arduino")


if __name__ == "__main__":
    reader = ArduinoReader(port='COM7')
    if reader.connect():
        try:
            while True:
                data = reader.get_reading()
                print(
                    f"HR:{data['heart_rate']} "
                    f"TEMP:{data['body_temp']} "
                    f"SPO2:{data['spo2']} "
                    f"ACT:{data['activity_level']} "
                    f"AMB:{data['ambient_temp']} "
                    f"HUM:{data['humidity']} "
                    f"HRSTATUS:{data['hr_status']}"
                )
                time.sleep(2)
        except KeyboardInterrupt:
            reader.disconnect()