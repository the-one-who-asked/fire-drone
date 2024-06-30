"""
Detects fires in IR images using a rule-based system which exhausts few of the drone's resources and is optimised using
the onboard GPU.

IR images are cleaned using temporal filtering and the fire detection method minimises false negatives so that frames
which are believed to show evidence of fire can be sent to a server with a more powerful detection algorithm.
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
SAIL_SEARCH = False
FIRE_THRESHOLD = 2

# Variables
delta_t = 0
speed = 1
theta = 0
d_theta = 0
img_speed = (SIZE[1] * speed) // (IMG_AREA[1] * FPS)
underlap = SIZE[1] // img_speed
frames = [np.zeros(SIZE, dtype=np.uint8) for _ in range(min(underlap, FPS))]  # Queue
underlaps = [SIZE[0]] * min(underlap, FPS)                                    # Queue

"====================================================== Modules ======================================================="
def get_frame() -> np.ndarray:
    """Captures a new frame from the NoIR camera"""

    return np.zeros(SIZE)


def update_speed():
    global speed, theta, d_theta, img_speed
    theta += d_theta * delta_t
    theta %= 360


def rotate_img(frame: np.ndarray, angular_velocity: float) -> np.ndarray:
    """Rotates an image through a certain angular velocity"""

    centre = ((SIZE[1] + 1) / 2 - angular_velocity / speed, (SIZE[0] + 1) / 2)
    rot_mat = cv2.getRotationMatrix2D(centre, angular_velocity, 1.)
    return cv2.warpAffine(frame, rot_mat, (SIZE[1], SIZE[0]), flags=cv2.INTER_LINEAR)


def clean_frame(frame: np.ndarray):
    """Removes noise from new frames through the use of temporal filtering"""

    # Initialises variables depending on whether the drone is rotating
    global underlaps
    prev_frame = frames[-1]
    silhouette = np.ones(SIZE)
    if SAIL_SEARCH:
        i = d_theta
        difference = lambda: np.sum(abs(rotate_img(prev_frame, i) - frame) * rotate_img(silhouette, i))
    else:
        i = img_speed
        difference = lambda: np.sum(abs(prev_frame[i:] - frame[:-i]))

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
    frames.pop(0)
    frames.append(frame)
    underlaps.pop(0)
    underlaps.append(0)
    for idx in range(len(underlaps)):
        underlaps[idx] += i

    # Gets the median pixel at regions of overlap, removing any noise and ensuring that fires detected are growing
    if SAIL_SEARCH:
        for i, (prev, current) in zip(underlaps, underlaps[1:] + [0]):
            mask = np.where(rotate_img(silhouette, current) != 0 and rotate_img(silhouette, prev) == 0)
            frame[mask] = frames[SIZE - i // -2][mask]
    else:
        for i, (prev, current) in zip(underlaps, underlaps[1:] + [0]):
            frame[prev:current] = frames[SIZE - i // -2][prev:current]


def fire_pixels(frame: np.ndarray) -> np.ndarray:
    """Finds pixels with relatively hight temperatures"""

    st_dev = np.std(frame, ddof=1)
    return frame > FIRE_THRESHOLD * np.std(frame, ddof=1)


def transmit(frame: np.ndarray, mask: np.ndarray):
    """Sends a frame to the server using a radio transmitter"""

    pass

"======================================================== Main ========================================================"
def main():
    global delta_t
    while True:
        delta_t = time()
        frame = get_frame()
        clean_frame(frame)
        mask = fire_pixels(frame)
        if mask.any():
            transmit(frame, mask)
        delta_t -= time()
        sleep(SPF - delta_t)
        update_speed()

if __name__ == "__main__":
    main()
