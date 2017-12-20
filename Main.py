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

videolist = ["resources/Easy/FrankUnderwood.mp4", "resources/Easy/JonSnow.mp4", "resources/Easy/MrRobot.mp4", "resources/MarquesBrownlee.mp4", "resources/TheMartian.mp4",
             "resources/Medium/LucianoRosso1.mp4", "resources/Medium/LucianoRosso2.mp4", "resources/Medium/LucianoRosso3.mp4",
             "resources/Hard/Joker.mp4", "resources/Hard/LeonardoDiCaprio.mp4", "resources/Trump.mp4"]

#Filter images
filterIm1 = plt.imread("resources/CatEars.png")



def run_face_replacement_no_exceptions(video1, video2, filter, out_filename):
    try:
        face_replacement.face_replacement(video1, video2, out_filename, filter, debug=False)
    except:
        return

filter = None
#filter = filterIm1
for i in range(0,5):
    for j in range(len(videolist)):
        if i != j:
            print "Video %d with video %d" % (i, j)
            video1 = imageio.get_reader(videolist[i])
            video2 = imageio.get_reader(videolist[j])
            f_name = "resources/OutputVideo_%d_%d.mp4" % (i, j)
            run_face_replacement_no_exceptions(video1, video2, filter, f_name)
