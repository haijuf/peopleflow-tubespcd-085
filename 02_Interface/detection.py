import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

class PeopleCounter:
    def __init__(self, max_capacity=None):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        self.count_in = 0
        self.count_out = 0
        self.max_capacity = max_capacity
        self.tracked_people = {}  # Format: {id: (centroid_x, centroid_y)}
        self.next_id = 1
        self.line_x = 400  # Garis imajiner vertikal

    def get_centroid(self, landmarks, frame_shape):
        """Menghitung centroid dari landmark untuk satu orang."""
        h, w, _ = frame_shape
        # Ambil koordinat bahu kiri dan pinggul kiri
        shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        cx = int((shoulder.x + hip.x) / 2 * w)
        cy = int((shoulder.y + hip.y) / 2 * h)
        return cx, cy

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        status = "Normal"
        is_full = False

        # Simpan posisi orang saat ini
        current_people = {}

        # Proses landmark jika ada
        if results.pose_landmarks:
            # Hitung centroid untuk pose yang terdeteksi
            cx, cy = self.get_centroid(results.pose_landmarks, frame.shape)

            # Berikan ID unik untuk orang
            person_id = None
            for pid, (px, py) in self.tracked_people.items():
                if distance.euclidean((cx, cy), (px, py)) < 100:
                    person_id = pid
                    break

            if person_id is None:
                person_id = self.next_id
                self.next_id += 1

            current_people[person_id] = (cx, cy)

            # Deteksi perlintasan garis vertikal
            prev_pos = self.tracked_people.get(person_id, (cx, cy))
            prev_x = prev_pos[0]

            current_count = self.count_in - self.count_out
            if prev_x <= self.line_x and cx > self.line_x:
                # Cek apakah kapasitas sudah penuh
                if self.max_capacity is None or current_count < self.max_capacity:
                    self.count_in += 1  # Orang bergerak ke kanan (masuk)
            elif prev_x >= self.line_x and cx < self.line_x:
                self.count_out += 1  # Orang bergerak ke kiri (keluar)

            # Gambar centroid dan ID
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"ID: {person_id}", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update tracked people
        self.tracked_people = current_people

        # Gambar garis imajiner vertikal
        cv2.line(frame, (self.line_x, 0), (self.line_x, frame.shape[0]), (255, 0, 0), 2)

        # Cek kapasitas
        current_count = self.count_in - self.count_out
        if self.max_capacity:
            if current_count >= self.max_capacity:
                is_full = True
                status = "Ruangan Penuh"
                cv2.putText(frame, "Ruangan Penuh!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            elif current_count == 0:
                status = "Ruangan Kosong"
            else:
                status = f"Sisa Kapasitas: {self.max_capacity - current_count}"
        else:
            if current_count == 0:
                status = "Ruangan Kosong"
            else:
                status = f"Total Orang: {current_count}"

        return frame, self.count_in, self.count_out, status, is_full 