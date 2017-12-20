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

videolist = ["resources/Medium/LucianoRosso1.mp4", "resources/Medium/LucianoRosso2.mp4", "resources/Medium/LucianoRosso3.mp4",
             "resources/Hard/Joker.mp4", "resources/Hard/LeonardoDiCaprio.mp4", "resources/Trump.mp4"]


video1 = imageio.get_reader("resources/Hard/LeonardoDiCaprio.mp4")
video2 = imageio.get_reader("resources/Easy/MrRobot.mp4")
f_name = "resources/OutputVideoRedo%d_%d.mp4" % (2, 1)
face_replacement.face_replacement(video1, video2, f_name, None, debug=False)

# out_filename = "resources/OutputVideo.mp4"
filterIm1 = plt.imread("resources/CatEars.png")

# video1 = imageio.get_reader(videolist[0])
# video2 = imageio.get_reader(videolist[1])
# face_replacement.face_replacement(video1, video2, out_filename, None, debug=False)



def run_face_replacement_no_exceptions(video1, video2, out_filename):
    try:
        face_replacement.face_replacement(video1, video2, out_filename, None, debug=False)
    except:
        return


for i in range(len(videolist)):
    # for j in range(len(videolist)):
        if i != 5:
            print "Video %d with video %d" % (i, 5)
            video1 = imageio.get_reader(videolist[i])
            video2 = imageio.get_reader(videolist[5])
            f_name = "resources/OutputVideoRedo%d_%d.mp4" % (i, 5)
            run_face_replacement_no_exceptions(video1, video2, f_name)
