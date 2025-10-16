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


# Fraction code for base error handling
def interpolate_point(p1, p2, fraction):
    """
    Returns a point along the line from p1 to p2 at given fraction.
    fraction=0 → p1
    fraction=1 → p2
    """
    x = int(p1[0] + (p2[0] - p1[0]) * fraction)
    y = int(p1[1] + (p2[1] - p1[1]) * fraction)
    return (x, y)

# Drawing roatating custom circles for jarvis !!
import time

def draw_rotating_custom_circle(img, center, radius, color, thickness, speed, clockwise, gaps=[]):
    """
    Draws a circle with gaps that rotates over time.
    speed: degrees per second
    clockwise: True for clockwise, False for anticlockwise
    """
    t = time.time()
    angle_offset = (speed * t) % 360
    if not clockwise:
        angle_offset = -angle_offset

    gaps = sorted(gaps, key=lambda x: x[0])
    current_angle = 0
    for gap_start, gap_end in gaps:
        # Rotate angles
        rotated_gap_start = (gap_start + angle_offset) % 360
        rotated_gap_end = (gap_end + angle_offset) % 360

        if rotated_gap_start < rotated_gap_end:
            cv2.ellipse(img, center, (radius, radius), 0,rotated_gap_start,rotated_gap_end, color, thickness, cv2.LINE_AA)
        else:
            # Arc wraps past 360° → split into two arcs
            cv2.ellipse(img, center, (radius, radius), 0,rotated_gap_start, 360, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, center, (radius, radius), 0, 0,rotated_gap_end, color, thickness, cv2.LINE_AA)
        
        """
            # Draw arc from current_angle to gap_start
            rotated_current = (current_angle + angle_offset) % 360
            if rotated_gap_start != rotated_current:
                cv2.ellipse(img, center, (radius, radius), 0, rotated_current, rotated_gap_start, color, thickness, cv2.LINE_AA)
            
            current_angle = gap_end

        # Draw final segment
        rotated_current = (current_angle + angle_offset) % 360
        if rotated_current != (0 + angle_offset) % 360:
            cv2.ellipse(img, center, (radius, radius), 0, rotated_current, (360 + angle_offset) % 360, color, thickness, cv2.LINE_AA)
        """
# Lines for open fingers
def draw_energy_conduit(img, points, base_color=(255, 240, 100), thickness=2, pulse_speed=3.0):
    """
    Draws elegant glowing energy lines with subtle motion and pulse.
    """

    t = time.time()

    # --- 1. Draw core path ---
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]

        # Gradient along the path
        for j in range(10):
            frac = j / 10
            interp_x = int(p1[0] + (p2[0] - p1[0]) * frac)
            interp_y = int(p1[1] + (p2[1] - p1[1]) * frac)
            alpha = int(255 * (1 - frac * 0.6))  # fade along length
            color = (int(base_color[0] * (alpha / 255)),
                     int(base_color[1] * (alpha / 255)),
                     int(base_color[2] * (alpha / 255)))
            cv2.circle(img, (interp_x, interp_y), thickness, color, -1, cv2.LINE_AA)

    # --- 2. Add a subtle glow around the line ---
    for blur_radius in [4, 8, 12]:
        overlay = img.copy()
        cv2.polylines(overlay, [np.array(points, dtype=np.int32)],
                      isClosed=False, color=base_color, thickness=blur_radius, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.05, img, 0.95, 0, img)

    # --- 3. Energy pulse traveling along path ---
    total_len = 0
    seg_lengths = []
    for i in range(len(points) - 1):
        l = math.hypot(points[i+1][0]-points[i][0], points[i+1][1]-points[i][1])
        seg_lengths.append(l)
        total_len += l

    if total_len > 0:
        pulse_pos = ((math.sin(t * pulse_speed) + 1) / 2) * total_len
        accum = 0
        for i in range(len(points) - 1):
            if accum + seg_lengths[i] >= pulse_pos:
                frac = (pulse_pos - accum) / seg_lengths[i]
                px = int(points[i][0] + (points[i+1][0] - points[i][0]) * frac)
                py = int(points[i][1] + (points[i+1][1] - points[i][1]) * frac)
                cv2.circle(img, (px, py), 6, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (px, py), 10, (255, 255, 180), 1, cv2.LINE_AA)
                break
            accum += seg_lengths[i]

# --- Sci-Fi Visualization Functions (Static Version) ---

def draw_static_circle(img, center, radius, color, thickness):
    """Draws a static glowing circle"""
    cv2.circle(img, center, radius, color, thickness, cv2.LINE_AA)

def draw_energy_path(img, points, color, thickness=2):
    """Draws a static multi-segment energy line following finger joints"""
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], color, thickness, cv2.LINE_AA)

def add_jarvis_overlay(img, hand_landmarks, finger_states):
    h, w, _ = img.shape

    # Palm center
    cx, cy, cz = get_palm_center(hand_landmarks.landmark)
    palm_center_px = (int(cx * w), int(cy * h))

    # Draw main JARVIS core (no pulse)
    # Replace static core circle
    gaps0 = [(10, 50), (80, 110), (150, 200), (240, 310), (350, 360)] #5
    draw_rotating_custom_circle(img, palm_center_px, radius=13, color=(255, 255, 200), thickness=4, gaps=gaps0, speed=40, clockwise=True)
    gaps1 = [(0, 80), (100, 130), (160, 200), (230, 300), (320, 360)] #5
    draw_rotating_custom_circle(img, palm_center_px, radius=20, color=(255, 255, 200), thickness=3, gaps=gaps1, speed=80, clockwise=True)
    gaps2 = [(0, 80), (100, 130), (160, 200), (230, 300)] #4
    draw_rotating_custom_circle(img, palm_center_px, radius=25, color=(255, 255, 150), thickness=2, gaps=gaps2, speed=50, clockwise=True)
    gaps3 = [(0, 80), (100, 130), (160, 200), (230, 300)] #4
    draw_rotating_custom_circle(img, palm_center_px, radius=30, color=(255, 255, 200), thickness=3, gaps=gaps3, speed=100, clockwise=False)
    gaps4 = [(0, 80), (100, 130), (160, 200), (230, 300), (320, 360)] #5
    draw_rotating_custom_circle(img, palm_center_px, radius=35, color=(255, 255, 100), thickness=2, gaps=gaps4, speed=50, clockwise=True)
    gaps5 = [(0, 100), (120, 190), (250, 320)] #3
    draw_rotating_custom_circle(img, palm_center_px, radius=41, color=(255, 255, 0), thickness=4, gaps=gaps5, speed=65, clockwise=False)
    
    #draw_static_circle(img, palm_center_px, radius=20, color=(255, 255, 200), thickness=3)
    #draw_static_circle(img, palm_center_px, radius=25, color=(255, 255, 0), thickness=2)
    #draw_static_circle(img, palm_center_px, radius=30, color=(255, 255, 0), thickness=3)
    #draw_static_circle(img, palm_center_px, radius=40, color=(255, 255, 0), thickness=2)
    #draw_static_circle(img, palm_center_px, radius=47, color=(255, 255, 0), thickness=4)

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
            draw_energy_conduit(img, all_points, base_color=(255, 240, 100), thickness=1, pulse_speed=1.5)
            #draw_energy_path(img, all_points, color=(200, 200, 50), thickness=1)

            # Draw small circle at fingertip
            tip_px = joint_coords[-1]
            gaps6 = [(0, 100), (120, 220), (240, 340)] #3
            draw_rotating_custom_circle(img, tip_px, radius=20, color=(255, 255, 175), thickness=3, gaps=gaps6, speed=65, clockwise=False)
            gaps7 = [(0, 360)] #complete
            draw_rotating_custom_circle(img, tip_px, radius=0, color=(0, 100, 100), thickness=4, gaps=gaps7, speed=65, clockwise=False)
            gaps8 = [(0, 360)] #complete
            draw_rotating_custom_circle(img, tip_px, radius=5, color=(255, 255, 175), thickness=1, gaps=gaps8, speed=65, clockwise=False)
    

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
            add_jarvis_overlay(img, hand_landmarks, finger_states)

    cv2.imshow("JARVIS Hand Tracker", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
