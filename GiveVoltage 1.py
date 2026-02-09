 
import serial
import time
import keyboard
 
arduino_port = 'COM3'
baud_rate = 9600
 
# Connect to Arduino on COM3 (adjust baudrate if needed)
try:
    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    print(f"Connected to Arduino on {arduino_port}")
except serial.SerialException:
    print("Could not connect. Is the Arduino connected?")
    exit()
 
space_pressed = False
 
print("Press space to send a signa to the Arduino. Press ESC to quit.")
 
while True:
    try:
        if keyboard.is_pressed('space'):
            if not space_pressed:
                ser.write(b'C')
                print("Signal Sent")
                space_pressed = True
        else:
            space_pressed = False

        if keyboard.is_pressed('esc'):
            print("Exiting...")
            break
              
        time.sleep(0.1)
    except Exception as e:
        print(f"Error occured: {e}")
        break
 
ser.close()