from zipfile import ZipFile

import cv2 as cv
import pytesseract
from PIL import Image, ImageDraw

# loading the face detection classifier
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# processing zip
InPath = "small_img.zip"
OutPath = "images"


# Unzip and get names and path's to extracted images

def unzip(InPath=InPath, OutPath=OutPath):
    with ZipFile(InPath, 'r') as zipObj:
        zipObj.extractall(OutPath)


def getImageNames(InPath=InPath):
    with ZipFile(InPath, "r") as zipObj:
        names = zipObj.namelist()
        return names


def generatePath(OutPath=OutPath, names=getImageNames()):
    path = [OutPath + "/" + name for name in names]
    return path


def preparationToOCR(path=generatePath(), names=getImageNames()):
    objects = {}
    for i in range(len(names)):
        objects = {names[i]: Image.open(path[i]).convert("1")}
    return objects


def getTextFromPage(items=preparationToOCR()):
    parsedPages = {}
    for key in preparationToCV().keys():
        parsedPages = {
            key: pytesseract.image_to_string(preparationToOCR()[key])}
    return parsedPages


def preparationToCV(path=generatePath(), names=getImageNames()):
    object = {}
    for i in range(len(names)):
        img = cv.imread(path[i])
        object = {names[i]: cv.cvtColor(img, cv.COLOR_BGR2GRAY)}
        return object


def getFacesFromPage(items=preparationToCV(), images=preparationToOCR()):
    faces = {}
    for key in preparationToCV().keys():
        faces = {key: face_cascade.detectMultiScale(preparationToCV()[key])}

    def show_rects(items=preparationToCV(), images=preparationToOCR()):
        headsFromImage = {}
        for key in preparationToCV().keys():
            pil_img = preparationToOCR()[key].convert("RGB")
            drawing = ImageDraw.Draw(pil_img)
            for x, y, w, h in faces[key]:
                drawing.rectangle((x, y, x + w, y + h), outline="red")
                headsFromImage = {key: pil_img}
        return headsFromImage

    return show_rects()


for val in getFacesFromPage().values():
    val.show()