import cv2
import numpy as np
import amcam


class EmbryoDetector:
    def __init__(self):
        self.hcam = None
        self.running = True

    # Event callback — correct signature for Amcam SDK
    @staticmethod
    def _event_callback(nEvent, pCtx):
        detector = pCtx
        if nEvent == amcam.AMCAM_EVENT_IMAGE:
            try:
                # Pull latest frame when image event is triggered
                frame = detector.hcam.PullImageV3()
                if frame is not None:
                    detector._process_frame(frame)
            except Exception as e:
                print(f"Frame pull error: {e}")

    # Frame processor (your image analysis pipeline goes here)
    def _process_frame(self, frame):
        # Convert to grayscale if camera outputs color
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes for potential embryos
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < 5000:  # tune thresholds for embryo size
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show result
        cv2.imshow("Embryo Detector", display)

        # ESC key to stop
        if cv2.waitKey(1) & 0xFF == 27:
            self.running = False

    def run(self):
        # Open first connected camera
        self.hcam = amcam.Amcam.Open(None)
        if not self.hcam:
            print("No camera found")
            return

        # Print camera info
        # info = self.hcam.get_DevInfo()
        # print(f"Connected to: {info.displayname}")
        # print(f"Serial: {info.id.decode()}")
        # print(f"Resolution: {self.hcam.res[0]}x{self.hcam.res[1]}")

        # Start camera in pull mode with callback
        self.hcam.StartPullModeWithCallback(EmbryoDetector._event_callback, self)

        print("Press ESC in window to stop...")
        while self.running:
            cv2.waitKey(10)

        # Cleanup
        self.hcam.Close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = EmbryoDetector()
    detector.run()
