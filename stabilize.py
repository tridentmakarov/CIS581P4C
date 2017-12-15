import face_replacement as face_rep
import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt

def stabilize(source, output, nFrames):



    frame1 = source.get_next_data()
    frame1Gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    bbox = face_rep.detect_faces(frame1)

    x = bbox[0, 0]
    y = bbox[0, 1]
    w = bbox[0, 2]
    h = bbox[0, 3]

    bboxPolygon = [x, y, x + w, y, x + w,  y + h, x, y + h]
    plotbboxPolygon = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    pts = plotbboxPolygon.reshape((-1, 1, 2))
    print "Face Detected"

    videoFrame = cv2.polylines(frame1,[pts],True,(0,255,255))

    # plt.imshow(videoFrame)
    # plt.show()

    points = cv2.cornerHarris(frame1Gray[x: x + w, y: y + h], 3, 3, 0.04)

    plt.imshow(videoFrame)
    #plt.show()