'''

Main File, here to combine the main functions of the code.
1) Haar Cascades

2) Feature Detection
3) Stabilization
4) Alignment

'''
from PIL import Image
from matplotlib import pyplot as plt
import face_replacement as face_rep
import imageio
import face_replacement
from stabilize import stabilize


video1 = imageio.get_reader("resources/MarquesBrownlee.mp4")
video2 = imageio.get_reader("resources/TheMartian.mp4")

stabilize(video1, "resources/stabilized", 74)

face_replacement.face_replacement(video1, video2)

outVideo = imageio.get_writer("resources/testOutput.mp4", fps=video1._meta['fps'])

nFrames = video1._meta['nframes']
prev_frame = video1.get_next_data()

cascade = face_rep.detect_faces(prev_frame)

