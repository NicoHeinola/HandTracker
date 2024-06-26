from threading import Thread
import cv2
import pyvirtualcam
from hand_detector import HandDetector


class VirtualHandWebcam:
    def __init__(self) -> None:
        self._debug: bool = False
        self.camera_capture_id: int = 0
        self.fps: int = 20

        self._window_thread: Thread = None
        self.running: bool = False

        self._hand_detector: HandDetector = HandDetector()

        self._current_hand_frame = None
        self._current_full_frame = None

    def set_hand_tracking_accuracy(self, min_detection_confidence: float, min_tracking_confidence: float) -> None:
        self._hand_detector._min_detection_confidence = min_detection_confidence
        self._hand_detector._min_tracking_confidence = min_tracking_confidence

    def set_desired_size(self, width: int, height: int) -> None:
        self._hand_detector.set_desired_size(width, height)

    def set_transition_percentage(self, transition_percentage: float) -> None:
        self._hand_detector._transition_percentage = transition_percentage

    def set_transition_type(self, transition_type: str) -> None:
        self._hand_detector._transition_type = transition_type

    def set_transition_distance_min(self, transition_distance_min) -> None:
        self._hand_detector._transition_distance_min = transition_distance_min

    def set_minimum_hand_movement(self, minimum_hand_movement) -> None:
        self._hand_detector._minimum_hand_movement = minimum_hand_movement

    def set_debug(self, isDebug: bool) -> None:
        self._debug = isDebug
        self._hand_detector.debug = isDebug

    def stop_webcam(self):
        self.running = False

    def _start_debug_windows(self) -> None:
        while self.running:
            if self._current_hand_frame is None and self._current_full_frame is None:
                continue

            if self._current_full_frame is not None:
                cv2.imshow('Full capture', self._current_full_frame)

            if self._current_hand_frame is not None:
                cv2.imshow('Hand only', self._current_hand_frame)

            # Check for 'q' key press to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def start_webcam(self) -> None:
        self.running = True

        if self._debug:
            self._window_thread = Thread(target=self._start_debug_windows, daemon=True)
            self._window_thread.start()

        with pyvirtualcam.Camera(width=self._hand_detector.desired_width, height=self._hand_detector.desired_height, fps=self.fps) as cam:
            print(f'Created a virtual camera: {cam.device}')
            print("Connecting to a camera... Please wait.")

            cap: cv2.VideoCapture = cv2.VideoCapture(self.camera_capture_id)

            print("Connected to the camera!")
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Get the new frame
                self._hand_detector.process_image(frame)
                self._hand_detector.animate_image()
                self._current_hand_frame = self._hand_detector.get_image()

                # Save the full frame
                if self._debug:
                    cv2.rectangle(frame, (self._hand_detector._last_crop_x1, self._hand_detector._last_crop_y1), (self._hand_detector._last_crop_x2, self._hand_detector._last_crop_y2), (255, 0, 0), 2)
                    self._current_full_frame = frame

                cam.send(self._current_hand_frame)
                cam.sleep_until_next_frame()

            cap.release()
            self.running = False
