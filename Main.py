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

videolist = ["resources/MarquesBrownlee.mp4", "resources/TheMartian.mp4", "resources/Easy/FrankUnderwood.mp4", "resources/Easy/MrRobot.mp4", "resources/Easy/JonSnow.mp4",
             "resources/Medium/LucianoRosso1.mp4", "resources/Medium/LucianoRosso2.mp4", "resources/Medium/LucianoRosso3.mp4",
             "resources/Hard/Joker.mp4", "resources/Hard/LeonardoDiCaprio.mp4"]


out_filename = "resources/OutputVideo.mp4"
filterIm1 = plt.imread("resources/CatEars.png")



def run_face_replacement_no_exceptions(video1, video2, out_filename):
    try:
        face_replacement.face_replacement(video1, video2, out_filename, None, debug=False)
    except:
        return


for i in range(len(videolist)):
    for j in range(i+1, len(videolist)):
        video1 = imageio.get_reader(videolist[i])
        video2 = imageio.get_reader(videolist[j])
        f_name = "resources/OutputVideo%d_%d.mp4" % (i, j)
        run_face_replacement_no_exceptions(video1, video2, f_name)