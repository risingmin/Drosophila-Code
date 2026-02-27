import serial

ser = serial.Serial('COM3', 9600)
print("Press Enter to send signal.")

while True:
    input()
    ser.write(b'C')
    print("Signal Sent")