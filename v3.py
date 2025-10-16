import cv2
import mediapipe as mp
import math
import time
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Utility functions
def get_palm_center(landmarks):
    indices = [0, 1, 5, 9, 13, 17]
    x = sum(landmarks[i].x for i in indices) / len(indices)
    y = sum(landmarks[i].y for i in indices) / len(indices)
    z = sum(landmarks[i].z for i in indices) / len(indices)
    return x, y, z

def euclidean_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def get_finger_states(hand_landmarks):
    finger_states = []

    # Palm center
    indices = [0, 1, 5, 9, 13, 17]
    center = hand_landmarks.landmark
    cx = sum(center[i].x for i in indices) / len(indices)
    cy = sum(center[i].y for i in indices) / len(indices)
    cz = sum(center[i].z for i in indices) / len(indices)

    class Point:
        def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z
    palm_center = Point(cx, cy, cz)

    # Thumb
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ref = hand_landmarks.landmark[2]
    dist_tip = euclidean_distance(thumb_tip, palm_center)
    dist_ref = euclidean_distance(thumb_ref, palm_center)
    thumb_open = dist_tip > dist_ref
    if (thumb_tip.x < cx and thumb_ref.x > cx) or (thumb_tip.x > cx and thumb_ref.x < cx):
        thumb_open = False
    finger_states.append(1 if thumb_open else 0)

    # Other fingers
    finger_tips = [8, 12, 16, 20]
    finger_refs = [6, 10, 14, 18]
    for tip_idx, ref_idx in zip(finger_tips, finger_refs):
        tip = hand_landmarks.landmark[tip_idx]
        ref = hand_landmarks.landmark[ref_idx]
        dist_tip = euclidean_distance(tip, palm_center)
        dist_ref = euclidean_distance(ref, palm_center)
        finger_states.append(1 if dist_tip > dist_ref else 0)

    return finger_states


# fraction code for base error handling
def interpolate_point(p1, p2, fraction):
    """
    Returns a point along the line from p1 to p2 at given fraction.
    fraction=0 → p1
    fraction=1 → p2
    """
    x = int(p1[0] + (p2[0] - p1[0]) * fraction)
    y = int(p1[1] + (p2[1] - p1[1]) * fraction)
    return (x, y)

# --- Sci-Fi Visualization Functions (Static Version) ---

def draw_static_circle(img, center, radius, color, thickness):
    """Draws a static glowing circle"""
    cv2.circle(img, center, radius, color, thickness, cv2.LINE_AA)

def draw_energy_path(img, points, color, thickness=2):
    """Draws a static multi-segment energy line following finger joints"""
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], color, thickness, cv2.LINE_AA)

def add_jarvis_overlay(img, hand_landmarks, finger_states, line_thickness=2):
    h, w, _ = img.shape

    # Palm center
    cx, cy, cz = get_palm_center(hand_landmarks.landmark)
    palm_center_px = (int(cx * w), int(cy * h))

    # Draw main JARVIS core (no pulse)
    draw_static_circle(img, palm_center_px, radius=20, color=(255, 255, 200), thickness=3)
    draw_static_circle(img, palm_center_px, radius=25, color=(255, 255, 0), thickness=2)
    draw_static_circle(img, palm_center_px, radius=30, color=(255, 255, 0), thickness=3)
    draw_static_circle(img, palm_center_px, radius=40, color=(255, 255, 0), thickness=2)
    draw_static_circle(img, palm_center_px, radius=47, color=(255, 255, 0), thickness=4)

    # Define finger joint indices (base → mid → tip)
    finger_joint_chains = {
        0: [2, 4],       # Thumb
        1: [5, 8],       # Index
        2: [9, 12],     # Middle
        3: [13, 16],    # Ring
        4: [17, 20]     # Pinky
    }

    # Draw lines and finger tip markers for open fingers
    for i, state in enumerate(finger_states):
        if state == 1:  # Finger open
            # Get joint coordinates
            joint_coords = []
            for idx in finger_joint_chains[i]:
                lm = hand_landmarks.landmark[idx]
                joint_coords.append((int(lm.x * w), int(lm.y * h)))

            # Determine line start from outer JARVIS circle
            first_joint = joint_coords[0]
            dir_x, dir_y = first_joint[0] - palm_center_px[0], first_joint[1] - palm_center_px[1]
            mag = math.sqrt(dir_x**2 + dir_y**2)
            if mag != 0:
                dir_x, dir_y = dir_x / mag, dir_y / mag
            outer_circle_point = (int(palm_center_px[0] + dir_x * 47),
                                  int(palm_center_px[1] + dir_y * 47))

            # Adjust base point along finger (fraction toward tip, e.g., 20%)
            base_point = interpolate_point(joint_coords[0], joint_coords[-1], 0.2)

            # Combine points: outer circle → adjusted base → rest of finger joints
            all_points = [outer_circle_point, base_point] + joint_coords[1:]

            # Draw connecting line through all points
            draw_energy_path(img, all_points, color=(200, 200, 50), thickness=1)

            # Draw small circle at fingertip
            tip_px = joint_coords[-1]
            draw_static_circle(img, tip_px, radius=20, color=(255, 225, 0), thickness=1)

# --- Main Capture Loop ---
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            finger_states = get_finger_states(hand_landmarks)
            #mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw JARVIS overlay
            add_jarvis_overlay(img, hand_landmarks, finger_states, line_thickness=3)

    cv2.imshow("JARVIS Hand Tracker", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
