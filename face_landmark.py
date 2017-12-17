import dlib

predictor_path = "resources/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

def get_face_landmarks(img, rect):
    (x,y, w, h) = rect
    dlib_rect = [(x,y), (x + w, y + h)]
    shape = predictor(img, dlib_rect)
    return [(p.x, p.y) for p in shape.parts()]