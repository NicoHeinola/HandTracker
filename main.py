
from virtual_hand_webcam import VirtualHandWebcam


if __name__ == "__main__":
    virtual_hand_webcam: VirtualHandWebcam = VirtualHandWebcam()

    # ----------- SETTINGS
    # If you want to debug the webcam image
    virtual_hand_webcam.set_debug(True)

    # What resolution you want the webcam to be
    virtual_hand_webcam.set_desired_size(300, 300)

    # How accurate do you want the ai to be when tracking your hand. First value means finding accuracy of hand and second tracking accuracy. These should be low especially if wearing gloves.
    virtual_hand_webcam.set_hand_tracking_accuracy(0.1, 0.0)

    # When hand position changes, we need to move the camera to a new position of x amount. This determines how many percentages of that x amount we move at once. 1 = instant.
    virtual_hand_webcam.set_transition_percentage(0.0008)

    # What kind of transition you want the camera to do to the next hand position
    # halving: distance multiplied by percentage every loop
    # exponential: speed gets faster and faster the closer we are to the new hand position
    # sigmoid: speed gets slower and slower the closer we are to the new hand position
    virtual_hand_webcam.set_transition_type("sigmoid")

    # How much distance from the current position to the desired position should "snap" the current position to the desired position
    virtual_hand_webcam.set_transition_distance_min(15)

    # How much at minimum the hand must move from it's original position for us to start following it
    virtual_hand_webcam.set_minimum_hand_movement(5)

    # Desired webcam id (depends on how many webcams you have)
    virtual_hand_webcam.camera_capture_id = 1

    # What fps this webcam runs at
    virtual_hand_webcam.fps = 30
    # ----------- SETTINGS

    virtual_hand_webcam.start_webcam()
