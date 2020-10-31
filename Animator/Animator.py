import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from tkinter import Tk, Frame, LEFT, Label, BOTH, GROOVE, Button, filedialog, PhotoImage, messagebox

import numpy as np
import PIL.Image
import PIL.ImageTk
import cv2
import dlib
import torch

from poser.morph_rotate_combine_poser import MorphRotateCombinePoser256Param6
from puppet.head_pose_solver import HeadPoseSolver
from poser.poser import Poser
from puppet.util import compute_left_eye_normalized_ratio, compute_right_eye_normalized_ratio, \
    compute_mouth_normalized_ratio
from tha.combiner import CombinerSpec
from tha.face_morpher import FaceMorpherSpec
from tha.two_algo_face_rotator import TwoAlgoFaceRotatorSpec
from util import rgba_to_numpy_image, extract_pytorch_image_from_filelike

from msvcrt import getch

from database import *

class Animator:
    
    def __init__(self,
                 poser: Poser,
                 face_detector,
                 landmark_locator,
                 video_capture,
                 torch_device: torch.device):
        self.database=DataLoarder().create_database()

        self.poser = poser
        self.face_detector = face_detector
        self.landmark_locator = landmark_locator
        self.video_capture = video_capture
        self.torch_device = torch_device
        self.head_pose_solver = HeadPoseSolver()

        self.pose_size = len(self.poser.pose_parameters())
        self.source_image = None
        self.posed_image = None
        self.current_pose = None
        self.last_pose = None

        if not self.database.SettingImage == None:
            self.source_image=extract_pytorch_image_from_filelike(self.database.SettingImage).to(self.torch_device).unsqueeze(dim=0)


    def update_image(self):
        there_is_frame, frame = self.video_capture.read()
        if not there_is_frame:
            return
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.face_detector(rgb_frame)
        euler_angles = None
        face_landmarks = None
        if len(faces) > 0:
            face_rect = faces[0]
            face_landmarks = self.landmark_locator(rgb_frame, face_rect)
            face_box_points, euler_angles = self.head_pose_solver.solve_head_pose(face_landmarks)
            self.draw_face_landmarks(rgb_frame, face_landmarks)
            self.draw_face_box(rgb_frame, face_box_points)

        resized_frame = cv2.flip(cv2.resize(rgb_frame, (192, 256)), 1)
        pil_image = PIL.Image.fromarray(resized_frame, mode='RGB')



        if euler_angles is not None and self.source_image is not None:
            self.current_pose = torch.zeros(self.pose_size, device=self.torch_device)
            self.current_pose[0] = max(min(-euler_angles.item(0) / 15.0, 1.0), -1.0)
            self.current_pose[1] = max(min(-euler_angles.item(1) / 15.0, 1.0), -1.0)
            self.current_pose[2] = max(min(euler_angles.item(2) / 15.0, 1.0), -1.0)

            if self.last_pose is None:
                self.last_pose = self.current_pose
            else:
                self.current_pose = self.current_pose * 0.5 + self.last_pose * 0.5
                self.last_pose = self.current_pose

            eye_min_ratio = 0.15
            eye_max_ratio = 0.25
            left_eye_normalized_ratio = compute_left_eye_normalized_ratio(face_landmarks, eye_min_ratio, eye_max_ratio)
            self.current_pose[3] = 1 - left_eye_normalized_ratio
            right_eye_normalized_ratio = compute_right_eye_normalized_ratio(face_landmarks,
                                                                            eye_min_ratio,
                                                                            eye_max_ratio)
            self.current_pose[4] = 1 - right_eye_normalized_ratio

            min_mouth_ratio = 0.02
            max_mouth_ratio = 0.3
            mouth_normalized_ratio = compute_mouth_normalized_ratio(face_landmarks, min_mouth_ratio, max_mouth_ratio)
            self.current_pose[5] = mouth_normalized_ratio

            self.current_pose = self.current_pose.unsqueeze(dim=0)

            posed_image = self.poser.pose(self.source_image, self.current_pose).detach().cpu()
            numpy_image = rgba_to_numpy_image(posed_image[0])
            pil_image = PIL.Image.fromarray(np.uint8(np.rint(numpy_image * 255.0)), mode='RGBA')

            return pil_image


    def on_closing(self):
        self.video_capture.release()

    def draw_face_box(self, frame, face_box_points):
        line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                      [4, 5], [5, 6], [6, 7], [7, 4],
                      [0, 4], [1, 5], [2, 6], [3, 7]]
        for start, end in line_pairs:
            cv2.line(frame, face_box_points[start], face_box_points[end], (255, 0, 0), thickness=2)

    def draw_face_landmarks(self, frame, face_landmarks):
        for i in range(68):
            part = face_landmarks.part(i)
            x = part.x
            y = part.y
            cv2.rectangle(frame, (x - 1, y - 1), (x + 1, y + 1), (0, 255, 0), thickness=2)


def CreateAnimator():
    cuda = torch.device('cuda')
    poser = MorphRotateCombinePoser256Param6(
        morph_module_spec=FaceMorpherSpec(),
        morph_module_file_name="data/face_morpher.pt",
        rotate_module_spec=TwoAlgoFaceRotatorSpec(),
        rotate_module_file_name="data/two_algo_face_rotator.pt",
        combine_module_spec=CombinerSpec(),
        combine_module_file_name="data/combiner.pt",
        device=cuda)

    face_detector = dlib.get_frontal_face_detector()
    landmark_locator = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

    video_capture = cv2.VideoCapture(0)

    animator = Animator(poser, face_detector, landmark_locator, video_capture, cuda)
    return animator

if __name__ == "__main__":
    animator=CreateAnimator()

    while True:
        print("--- Ready OK !! ---")
        key = ord(getch())
        if key==13:
            animator.update_image().show()

