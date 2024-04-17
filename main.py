
from virtual_hand_webcam import VirtualHandWebcam


if __name__ == "__main__":
    virtual_hand_webcam: VirtualHandWebcam = VirtualHandWebcam()

    # ----------- SETTINGS
    # If you want to debug the webcam image
    virtual_hand_webcam.set_debug(True)

    # What resolution you want the webcam to be
    virtual_hand_webcam.set_desired_size(300, 300)

    # How accurate do you want the ai to be when tracking your hand. First value means finding accuracy of hand and second tracking accuracy. These should be low especially if wearing gloves.
    virtual_hand_webcam.set_hand_tracking_accuracy(0.15, 0.0)

    # Desired webcam id (depends on how many webcams you have)
    virtual_hand_webcam.camera_capture_id = 1

    # What fps this webcam runs at
    virtual_hand_webcam.fps = 24
    # ----------- SETTINGS

    virtual_hand_webcam.start_webcam()
