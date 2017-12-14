'''

Main File, here to combine the main functions of the code.
1) Haar Cascades

2) Feature Detection
3) Stabilization
4) Alignment

'''
from PIL import Image
from matplotlib import pyplot as plt
from extractFrames import extractFrames


img = Image.open("resources/barack-obama-eye-roll.gif")

plt.imshow(img[1])