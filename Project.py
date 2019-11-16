from zipfile import ZipFile

import cv2 as cv
import pytesseract
from PIL import Image

# loading the face detection classifier
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# processing zip
InPath = "small_img.zip"
OutPath = "images"


# Unzip and get names and path's to extracted images


def generateNamePathDict(InPath=InPath, OutPath=OutPath):
    namePath = {}
    with ZipFile(InPath, "r") as zipObj:
        zipObj.extractall(OutPath)
        names = zipObj.namelist()
        paths = [OutPath + "/" + name for name in names]
        for i in range(len(names)):
            namePath.update({names[i]: paths[i]})
    return namePath



# extract Text from pics
def preparationToOCR(items=generateNamePathDict()):
    objects = {key: Image.open(items[key]).convert("1") for key in items.keys()}

    return objects


def getTextFromPage(items=preparationToOCR()):
    parsedPages = {key: pytesseract.image_to_string(items[key]) for key in items.keys()}

    return parsedPages

# extract faces
def preparationToCV(items=generateNamePathDict()):
    object = {key: cv.cvtColor(cv.imread(items[key]), cv.COLOR_BGR2GRAY) for key in items.keys()}

    return object


def getFacesFromPage(items=preparationToCV()):
    faces = {key: face_cascade.detectMultiScale(items[key],
                                 scaleFactor=1.30,
                                 minNeighbors=5,
                                 minSize=(50, 50)) for key in items.keys()}
#was used to test image recognition, no longer neded
    # def show_rects(items=generateNamePathDict()):
    #     headsFromImage = {}
    #     for key in items.keys():
    #         with Image.open(items[key]).convert("RGB") as pil_img:
    #             drawing = ImageDraw.Draw(pil_img)
    #             for x, y, w, h in faces[key]:
    #                 drawing.rectangle((x, y, x + w, y + h), outline="red")
    #                 headsFromImage.update({key: pil_img})
    #     return headsFromImage

    def crop_faces(items = generateNamePathDict()):
        facesFromImage = {}
        for key in items.keys():
            with Image.open(items[key]).convert("RGB") as pil_img:
                for x, y, w, h in faces[key]:
                    listFaces = [pil_img.crop((x, y, x + w, y + h)) for x, y, w, h in faces[key]]
                facesFromImage[key]= listFaces
        return facesFromImage
    # return show_rects()
    return crop_faces()




def matchWordandFaces(word, text = getTextFromPage(), faces = getFacesFromPage()):
    for key in text.keys():
        if word in text[key]:
            result = {key:getFacesFromPage()[key]}
        else:result = f"{word} does not match any of pages"
    return result

if __name__ == '__main__':
    for val in getFacesFromPage().values():
        for i in range(len(val)):
            val[i].show()