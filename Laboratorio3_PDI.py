# This is a sample Python script.

from tkinter import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.preprocessing.image import img_to_array
from keras.models import model_from_json
from PIL import Image
from solver import resolve_sudoku

camara = cv2.VideoCapture(1)
debug=True
c=0
model = 0


class PySudoku(Frame):
    """Interfaz """

    def __init__(self, master, *args, **kwargs):
        """Interfaz """
        Frame.__init__(self, master, *args, **kwargs)
        self.parent = master
        self.grid()
        self.createWidgets()


    def createWidgets(self):  #cuadro de botones
        """Interfaz """

        self.camaraButton = Button(self, font=("Arial", 12), fg='red', text="Cámara", highlightbackground='red',
                               command=lambda: activarCam())
        self.camaraButton.grid(row=3, column=1, sticky="nsew")

        self.mostrarButton = Button(self, font=("Arial", 12), fg='red', text="Mostrar Foto", highlightbackground='lightgrey',
                                    command=lambda: leer_imagen())
        self.mostrarButton.grid(row=3, column=2, sticky="nsew")

        self.procesarButton = Button(self, font=("Arial", 12), fg='red', text="Procesar imagen", highlightbackground='lightgrey',
                                   command=lambda: proceImagen())
        self.procesarButton.grid(row=3, column=3, sticky="nsew")



def activarCam(): #activa la camara y con la tecla "F" toma una foto y la guarda como sudoku.png, con la tecla "s" cierra la camara
    """Interfaz """

    while (True):
        ok, frame = camara.read()
        cv2.rectangle(frame, (20, 20), (450, 450), (0, 255, 0))  # margen verde
        cv2.imshow('webCam',frame)

        if not ok:
            return False, None

        if (cv2.waitKey(1) == ord('s')):
             break
        if (cv2.waitKey(1) == ord('f')):

            frame = frame[21: 450, 21: 450]  #recorta la imagen al tamaño del margen verde
            cv2.imwrite('sudoku.png',frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break


def leer_imagen(): #muestra la foto tomada
    imagen = cv2.imread('sudoku.png')
    cv2.imshow('Imagen', imagen)


def proceImagen():
    """
    Entire process to solve the problem
    Prepare image for the DL processing
    Identify the numbers in the sudoku
    Create the sudoku in numpy
    Solve the sudoku
    """
    numbers = preprocesado()
    identify_images(numbers)
    sudoku = create_sudoku_matrix(numbers)
    resolve_sudoku(sudoku)

def preprocesado():
    """
    A lot of magic tricks to find those numbers
    :return: List of images custom objects with numbers of the sudoku and their coordinates
    """
    r = 1
    img = cv2.imread('sudoku.png', 0)
    im = cv2.imread('sudoku.png')
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))  # np.ones((7, 7), np.uint8)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)

    ret, binary = cv2.threshold(imgGray, 115, 255, cv2.THRESH_BINARY)
    ret, binary2 = cv2.threshold(imgBlur, 115, 255, cv2.THRESH_BINARY)
    ret, binary3 = cv2.threshold(imgGray, 100, 255, cv2.THRESH_BINARY)
    thresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    thresh = cv2.bitwise_not(thresh)
    thresh = cv2.bitwise_not(thresh)
    cierre = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # es la imagen final completamente limpia
    print(np.unique(cierre))
    cv2.imshow('Imagen cierre', cierre)

    contours, hierarchy = cv2.findContours(cierre, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # busca los contornos
    # contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    images = []
    numbers = []

    cv2.drawContours(im, contours, -1, (55, 254, 0), 5)  # muestra los contornos encima de la imagen cierre
    cv2.imshow("img", im)
    #cv2.imshow('Imagen thresh', thresh)
    #cv2.imshow('Imagen Blur', imgBlur)
    #cv2.imshow('Imagen GRAY', imgGray)
    #cv2.imshow('Imagen cierre', cierre)

    # We use the biggest contuour in the image and start finding its corners

    b = cv2.minAreaRect(contours[1])
    box = cv2.boxPoints(b)
    box = np.int0(box)
    im = cv2.drawContours(im, [box], 0, (0, 0, 255), 2)

    perimetro = cv2.arcLength(contours[1], True)
    dp = cv2.approxPolyDP(contours[1], perimetro * 0.02, True)
    if len(dp) == 4:
        cv2.drawContours(im, dp, -1, (255, 0, 0), 5)
        images.append(four_point_transform(cierre, dp.reshape(4, 2)))

        saves = False
        for i in images:
            numbers = cut_numbers(i)
            if not saves:
                cv2.imshow('numbers', numbers[0]['im'])
                saves = True
            plot_sudoku_images(numbers)

    return numbers

def cut_numbers(image):
    """
    Cuts the image in a 9x9 grid and pray for the numbers to fall in each cell
    :param image: Binarized sudoku image
    :return: Custom image object with coordinates and images of every number
    """
    numbers = []
    h = int(image.shape[1]/9)
    d = int(image.shape[0]/9)
    for i in range(9):
        for j in range(9):
            cut = (image[i*d:i*d+d, j*h: j*h+h])
            if not is_empty(cut_border(cut)):
                numbers.append({'i': i, 'j': j, 'im': cut_border(cut, 0.2)})
    return numbers


def plot_sudoku_images(images):
    """
    Plots sudoku images to check if there is numbers or not
    :param images:
    :return:
    """
    cols, rows = 9, 9
    fig = plt.figure(figsize=(9,9))
    grid = gridspec.GridSpec(ncols=cols,nrows=rows,figure=fig)

    for i in range(len(images)):
        fig.add_subplot(grid[i])
        label = f"Pos {images[i]['i']}-{images[i]['j']}"
        plt.title(label)
        plt.axis = False
        plt.imshow(images[i]['im'], cmap='binary')

    plt.show()


def is_empty(image, porcentaje=0.98):
    """
    Checks if the cell has a number or not
    :param image:
    :param porcentaje:
    :return:
    """

    border = int(image.shape[0] * 0.1)
    im = image[border: image.shape[0] - border, border: image.shape[1] - border]
    if cv2.countNonZero(im) > (im.shape[0] * im.shape[1] * porcentaje):
        return True
    return False


def cut_border(image, thickness=0.3):
    """
    Cuts a little bit of the margins in the image
    :param image: image to cut
    :param thickness: how muchs border will be cut
    :return: new image
    """
    d = int(image.shape[0] * thickness)
    h = int(image.shape[1] * thickness)
    return image[d : image.shape[0] - d, h : image.shape[1] - h]


def identify_images(image_objects):
    """ image_objects: list of images custom items
        images custom items are dictionarys with a coordinate i, a coordinate j and an image
    """

    img_arr = []
    for obj in image_objects:
        img = prepare_image(obj['im'])
        img_arr.append(img)

    img_arr = np.array(img_arr, dtype='float32')
    prediction = model.predict(img_arr)

    for index, value in enumerate(image_objects):
        value['prediction'] = (np.argmax(prediction[index]) + 1) % 10


def four_point_transform(image, pts):
    """"
    We took this two methods from internet
    """
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def prepare_image(image):
    """
    Prepare number images to enter to the predictor
    :param image:
    :return: preapred image
    """
    img = cv2.copyMakeBorder(image,5,5,5,5, cv2.BORDER_CONSTANT,value=(255,255))
    img = Image.fromarray(img)
    img = img_to_array(img.resize((96, 96)))
    img = img/255
    img = np.stack((img,) * 3, axis=-1).reshape(96,96,3)

    return img


def prepared_model():
    """
    Creates the model from training weights
    :return:
    """
    json_file = open('Digits_Recognizer_Model2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("Digit_Recognizer_Weights2")
    return model


def create_sudoku_matrix(image_objets):
    """
    Creates the numpy array with the sudoku to get ready for solving it.
    :param image_objets:
    :return:
    """
    sudoku = np.zeros((9,9))
    for obj in image_objets:
        sudoku[obj['i'], obj['j']] = obj['prediction']
    print(sudoku)
    return sudoku


Sudoku = Tk()
Sudoku.title("Sudoku Solver")
Sudoku.resizable(False, False)
root = PySudoku(Sudoku).grid()
model = prepared_model()
Sudoku.mainloop()
