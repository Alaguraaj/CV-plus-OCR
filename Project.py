from zipfile import ZipFile

import cv2 as cv
import pytesseract
from PIL import Image, ImageDraw

# loading the face detection classifier
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# processing zip
InPath = "small_img2.zip"
OutPath = "images"


# Unzip and get names and path's to extracted images


def generateNamePathDict(InPath=InPath, OutPath=OutPath):
    with ZipFile(InPath, "r") as zipObj:
        zipObj.extractall(OutPath)
        names = zipObj.namelist()
    paths = [OutPath + "/" + name for name in names]
    namePath = {name: path for name in names for path in paths}
    return namePath


# extract Text from pics
def preparationToOCR(items=generateNamePathDict()):
    objects = {key: Image.open(items[key]).convert("1") for key in items.keys()}

    return objects


def getTextFromPage(items=preparationToOCR()):
    parsedPages = {key: pytesseract.image_to_string(items()[key]) for key in items.keys()}

    return parsedPages

    # extract faces


def preparationToCV(items=generateNamePathDict()):
    object = {key: cv.cvtColor(cv.imread(items[key]), cv.COLOR_BGR2GRAY) for key in items.keys()}

    return object


def getFacesFromPage(items=preparationToCV()):
    faces = {key: face_cascade.detectMultiScale(items[key]) for key in items.keys()}

    def show_rects(items=generateNamePathDict()):
        headsFromImage = {}
        for key in items.keys():
            with Image.open(items[key]).convert("RGB") as pil_img:
                drawing = ImageDraw.Draw(pil_img)
                for x, y, w, h in faces[key]:
                    drawing.rectangle((x, y, x + w, y + h), outline="red")
                    headsFromImage.update({key: pil_img})
        return headsFromImage

    return show_rects()


if __name__ == '__main__':
    for pic in getFacesFromPage().values():
        pic.show()
#Same pic shows, need to investigate!!!!!!!!