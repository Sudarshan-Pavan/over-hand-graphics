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

# --- Sci-Fi Visualization Functions ---
def draw_pulsing_circle(img, center, base_radius, color, thickness=2, pulse_speed=2):
    """Draws a pulsing glowing circle"""
    t = time.time() * pulse_speed
    radius = int(base_radius + 3 * math.sin(t))
    cv2.circle(img, center, radius, color, thickness, cv2.LINE_AA)
    cv2.circle(img, center, radius + 5, color, 1, cv2.LINE_AA)

def draw_energy_line(img, start, end, color, pulse_speed=3):
    """Draws a glowing pulsing energy line"""
    t = abs(math.sin(time.time() * pulse_speed))
    overlay = img.copy()
    cv2.line(overlay, start, end, color, int(2 + 3 * t), cv2.LINE_AA)
    # The `line` function in the code is used to draw a glowing pulsing energy line on the image.
    # It takes the start and end points of the line, the color of the line, and a pulse speed
    # parameter that controls the pulsing effect of the line. The line is drawn with a varying
    # thickness based on the pulsing effect, creating a glowing energy line visual effect on the image.
    alpha = 0.5 + 0.5 * t
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def add_jarvis_overlay(img, hand_landmarks, finger_states):
    h, w, _ = img.shape

    # Palm center
    cx, cy, cz = get_palm_center(hand_landmarks.landmark)
    palm_center_px = (int(cx * w), int(cy * h))

    # Draw main JARVIS core
    draw_pulsing_circle(img, palm_center_px, base_radius=40, color=(255, 255, 200), thickness=2)
    draw_pulsing_circle(img, palm_center_px, base_radius=20, color=(255, 255, 0), thickness=2)

    # Draw finger tip circles + connect with energy lines
    finger_tips = [4, 8, 12, 16, 20]
    for i, state in enumerate(finger_states):
        if state == 1:  # Finger open
            tip = hand_landmarks.landmark[finger_tips[i]]
            tip_px = (int(tip.x * w), int(tip.y * h))
            draw_pulsing_circle(img, tip_px, base_radius=17, color=(255, 225, 100), thickness=1)
            draw_energy_line(img, palm_center_px, tip_px, color=(220, 100, 0))

    # Optional: Add faint holographic HUD rings around palm
    #for r in range(60, 140, 30):
    #    draw_pulsing_circle(img, palm_center_px, base_radius=r, color=(255, 255, 255), thickness=1, pulse_speed=0.5)

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
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw JARVIS overlay
            add_jarvis_overlay(img, hand_landmarks, finger_states)

    cv2.imshow("JARVIS Hand Tracker", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
