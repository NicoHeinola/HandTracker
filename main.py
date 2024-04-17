
from virtual_hand_webcam import VirtualHandWebcam


if __name__ == "__main__":
    virtual_hand_webcam: VirtualHandWebcam = VirtualHandWebcam()
    virtual_hand_webcam.set_debug(False)
    virtual_hand_webcam.set_desired_size(300, 300)
    virtual_hand_webcam.camera_capture_id = 1
    virtual_hand_webcam.fps = 60
    virtual_hand_webcam.start_webcam()
