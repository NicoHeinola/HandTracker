import math
import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    def __init__(self) -> None:
        self._hand_image = None
        self._processed_image = None

        self.desired_width: int = 300
        self.desired_height: int = 300

        # What position we are currently at on the hand cam
        self._last_crop_x1: int = 0
        self._last_crop_x2: int = self.desired_width
        self._last_crop_y1: int = 0
        self._last_crop_y2: int = self.desired_height

        # What position we want to be at on the hand cam
        self._destination_crop_x1: int = 0
        self._destination_crop_x2: int = self.desired_width
        self._destination_crop_y1: int = 0
        self._destination_crop_y2: int = self.desired_height

        # How "picky" the hand tracking ai is
        self._min_detection_confidence: float = 0.1
        self._min_tracking_confidence: float = 0

        self.debug: bool = False

        # Transition related
        self._transition_percentage: float = 0.5
        self._transition_type: str = "halving"  # Available types: halving, exponential, sigmoid
        self._transition_distance_min: float = 10
        self._minimum_hand_movement: float = 5

    def set_desired_size(self, width: int, height: int):
        self.desired_width = width
        self.desired_height = height

        self._last_crop_x1: int = 0
        self._last_crop_x2: int = self.desired_width
        self._last_crop_y1: int = 0
        self._last_crop_y2: int = self.desired_height

        self._destination_crop_x1: int = 0
        self._destination_crop_x2: int = self.desired_width
        self._destination_crop_y1: int = 0
        self._destination_crop_y2: int = self.desired_height

    def get_bounding_box(self, hand, area_width: int, area_height: int):
        """
        Finds a rectangular area around the hand

        :param hand: Hand to make the rectangle around of
        :param area_width: Width of the image
        :param area_height: Height of the image
        :return: Returns coordinates that represent a rectangle
        """

        # Rectangular position around the hand
        x_max, y_max = -float('inf'), -float('inf')
        x_min, y_min = float('inf'), float('inf')

        for landmark in hand.landmark:
            x, y = int(landmark.x * area_width), int(landmark.y * area_height)
            x_max = max(x_max, x)
            x_min = min(x_min, x)
            y_max = max(y_max, y)
            y_min = min(y_min, y)

        return x_min, x_max, y_min, y_max

    def detect_hands(self, image, draw_image: bool = True):
        """
        Detects hands from image

        :param image: Image to use to detect hands
        :param draw_image: Should we draw on top of given image. (Debug purposes)
        :return: Returns drawn image and results
        """

        # We want to find hands
        mp_hands = mp.solutions.hands

        # Try to find hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=self._min_detection_confidence, min_tracking_confidence=self._min_tracking_confidence)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Release resources
        hands.close()

        # If we found any hands, draw image of them to demonstrate
        if results.multi_hand_landmarks and draw_image:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return image, results

    def move_into_big_rect(self, big_rect, small_rect):
        big_x1, big_y1, big_x2, big_y2 = big_rect
        small_x1, small_y1, small_x2, small_y2 = small_rect

        # Check if small_rect is partially outside big_rect
        if (small_x1 < big_x1 and small_x2 > big_x1) or (small_y1 < big_y1 and small_y2 > big_y1) or \
                (small_x2 > big_x2 and small_x1 < big_x2) or (small_y2 > big_y2 and small_y1 < big_y2):

            # Calculate the amount of overlap on each side
            overlap_left = max(0, big_x1 - small_x1)
            overlap_top = max(0, big_y1 - small_y1)
            overlap_right = max(0, small_x2 - big_x2)
            overlap_bottom = max(0, small_y2 - big_y2)

            # Calculate the amount to move small_rect by
            move_x = overlap_left - overlap_right
            move_y = overlap_top - overlap_bottom

            # Update small_rect's position
            small_x1 += move_x
            small_y1 += move_y
            small_x2 += move_x
            small_y2 += move_y

            # Return the updated small_rect
            return (small_x1, small_y1, small_x2, small_y2)

        # If small_rect is completely inside big_rect, return its original position
        return (small_x1, small_y1, small_x2, small_y2)

    def generate_image_of_hand(self, original_image):
        self._processed_image, results = self.detect_hands(original_image, self.debug)

        # If no hands were detected
        if not results.multi_hand_landmarks:
            # If there were no hands, crop it from the last position
            cropped_image = self._processed_image[self._last_crop_y1:self._last_crop_y2, self._last_crop_x1:self._last_crop_x2]
            self._hand_image = cropped_image
            return

        image_width = self._processed_image.shape[1]
        image_height = self._processed_image.shape[0]

        # Get the hand bounding box
        main_hand = results.multi_hand_landmarks[0]
        x_min, x_max, y_min, y_max = self.get_bounding_box(main_hand, image_width, image_height)

        # Calculate center of the hand
        hand_center_x: int = int(round((x_min + x_max) / 2))
        hand_center_y: int = int(round((y_min + y_max) / 2))

        # Calculate crop box coordinates
        crop_x1 = int(hand_center_x - self.desired_width / 2)
        crop_y1 = int(hand_center_y - self.desired_height / 2)
        crop_x2 = int(hand_center_x + self.desired_width / 2)
        crop_y2 = int(hand_center_y + self.desired_height / 2)

        crop_x1, crop_y1, crop_x2, crop_y2 = self.move_into_big_rect((0, 0, image_width, image_height), (crop_x1, crop_y1, crop_x2, crop_y2))

        # Let's determine if the hand has moved enough from the previous destination
        destination_center_x = (self._destination_crop_x1 + self._destination_crop_x2) / 2
        destination_center_y = (self._destination_crop_y1 + self._destination_crop_y2) / 2
        new_destination_center_x = (crop_x1 + crop_x2) / 2
        new_destination_center_y = (crop_y1 + crop_y2) / 2

        distance = math.sqrt((new_destination_center_x - destination_center_x)**2 + (new_destination_center_y - destination_center_y)**2)

        if (distance < self._minimum_hand_movement):
            return

        # If hand has moved enough, we can move the rectangle that tracks it
        self._destination_crop_x1 = crop_x1
        self._destination_crop_x2 = crop_x2
        self._destination_crop_y1 = crop_y1
        self._destination_crop_y2 = crop_y2

    def process_image(self, image):
        self.generate_image_of_hand(image)

    def get_image(self):
        return self._hand_image

    def animate_image(self):

        percentage = 1

        last_center_x = (self._last_crop_x1 + self._last_crop_x2) / 2
        last_center_y = (self._last_crop_y1 + self._last_crop_y2) / 2
        destination_center_x = (self._destination_crop_x1 + self._destination_crop_x2) / 2
        destination_center_y = (self._destination_crop_y1 + self._destination_crop_y2) / 2
        distance = math.sqrt((destination_center_x - last_center_x)**2 + (destination_center_y - last_center_y)**2)

        if self._transition_type == "linear":
            percentage = self._transition_percentage
        elif self._transition_type == "exponential":
            percentage = math.exp(-self._transition_percentage * distance)
        elif self._transition_type == "sigmoid":
            percentage = 1 / (1 + math.exp(-self._transition_percentage * distance))

        if distance < self._transition_distance_min:
            percentage = 1

        self._last_crop_x1 += round((self._destination_crop_x1 - self._last_crop_x1) * percentage)
        self._last_crop_x2 += round((self._destination_crop_x2 - self._last_crop_x2) * percentage)
        self._last_crop_y1 += round((self._destination_crop_y1 - self._last_crop_y1) * percentage)
        self._last_crop_y2 += round((self._destination_crop_y2 - self._last_crop_y2) * percentage)

        # Crop the processed image
        cropped_image = self._processed_image[self._last_crop_y1:self._last_crop_y2, self._last_crop_x1:self._last_crop_x2]
        self._hand_image = cropped_image
