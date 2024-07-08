"""
Detects fires in images using a rule-based system which will be optimised using the onboard GPU. Frames are cleaned
using temporal filtering. The fire detection method minimises false negatives so that frames which are believed to show
evidence of fire can be sent to a server with a more powerful detection algorithm.
"""

"====================================================== Imports ======================================================="
import numpy as np
import cv2
from time import sleep, time

"===================================================== Variables ======================================================"
# Constants
SIZE = (1080, 1920)
FPS = 30
SPF = 1/FPS
IMG_AREA = (9, 16)
SAIL_SEARCH = False  # Indicates whether the drone is rotating freely or is moving in straight lines
Y_THRESHOLD = 1.5
CR_MIN, CR_MAX = 135, 180
CB_MIN, CB_MAX = 85, 135

# Variables
speed = 1
theta = 0
d_theta = 0
img_speed = (SIZE[1] * speed) // (IMG_AREA[1] * FPS)
underlap = SIZE[1] // img_speed
frames = np.zeros((min(underlap, FPS),) + SIZE, dtype=np.int8)  # Queue

"====================================================== Modules ======================================================="
def get_frame() -> np.ndarray:
    """Captures a new frame from the NoIR camera"""

    # capture frame in YUV
    # convert from YUV to YCbCr
    return np.zeros(SIZE)


def update_speed():
    """Modifies velocity and angular velocity based on sensor readings"""

    global speed, theta, d_theta, img_speed
    theta += d_theta * SPF
    theta %= 360


def rotate_img(frame: np.ndarray, angular_velocity: float) -> np.ndarray:
    """Rotates an image through a certain angular velocity"""

    centre = ((SIZE[1] + 1) / 2 - angular_velocity / speed, (SIZE[0] + 1) / 2)
    rot_mat = cv2.getRotationMatrix2D(centre, angular_velocity, 1.)
    return cv2.warpAffine(frame, rot_mat, (SIZE[1], SIZE[0]), flags=cv2.INTER_LINEAR)


def clean_frame(frame: np.ndarray) -> np.ndarray:
    """Removes noise from new frames through the use of temporal filtering"""

    # Initialises variables depending on whether the drone is rotating
    prev_frame = frames[-1]
    if SAIL_SEARCH:
        i = d_theta
        difference = lambda: np.sum(((mask := rotate_img(prev_frame, i)) - frame[mask]) ** 2)
    else:
        i = img_speed
        difference = lambda: np.sum((prev_frame[i:] - frame[:-i]) ** 2)

    # Maximises the overlap between consecutive frames
    current = difference()
    i -= 1
    below = difference()
    i += 2
    above = difference()
    if current > below or current > above:
        if above < below:
            step = 1
            current = above
            i += 1
        else:
            step = -1
            current = below
            i -= 3

        # Searches through velocities to find a local minimum in the underlap for a given velocity
        while i:
            prev = current
            current = difference()
            if current > prev:
                i -= step
                break
            i += step

    # Updates the queues of underlaps and frames to reflect the overlap between each frame and the current one
    frames[:-1] = frames[1:]
    if SAIL_SEARCH:
        for j, frame in enumerate(frames):
            frames[j] = rotate_img(frame, -i)
    else:
        frames[:, i:] = 0
    frames[-1] = frame

    # Gets the median pixel at regions of overlap, removing any noise and ensuring that fires detected are persistent
    return np.median(frames[:, frames[0].astype(bool)], axis=0)


def fire_pixels(frame: np.ndarray) -> np.ndarray:
    """Finds pixels with relatively high intensities and red components"""

    y, cr, cb = cv2.split(frame)
    return (y > np.mean(y) + Y_THRESHOLD * np.std(y, ddof=1)) and (CR_MIN <= cr <= CR_MAX) and (CB_MIN <= cb <= CB_MAX)


def transmit(frame: np.ndarray, mask: np.ndarray):
    """Sends a frame to the server using a radio transmitter"""

    print(frame)
    print(mask)

"======================================================== Main ========================================================"
def main():
    while True:
        delta_t = time()
        frame = get_frame()
        filtered = clean_frame(frame)
        mask = fire_pixels(filtered)
        if mask.any():
            transmit(frame, mask)
        delta_t -= time()
        sleep(SPF - delta_t)
        update_speed()

if __name__ == "__main__":
    main()
