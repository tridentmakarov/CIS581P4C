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



video1 = imageio.get_reader("resources/Easy/FrankUnderwood.mp4")
video2 = imageio.get_reader("resources/Easy/MrRobot.mp4")
out_filename = "resources/OutputVideo.mp4"
filterIm1 = plt.imread("resources/CatEars.png")

print "loaded"
face_replacement.face_replacement(video1, video2, out_filename, filterImg=None)
print "face"
outVideo = imageio.get_writer("resources/testOutput.mp4", fps=video1._meta['fps'])
print "out"

prev_frame = video1.get_next_data()
print "next"
cascade = face_rep.detect_faces(prev_frame)
print "done"
