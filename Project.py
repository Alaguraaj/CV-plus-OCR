from zipfile import ZipFile

import cv2 as cv
import numpy as np
import pytesseract
from PIL import Image

# loading the face detection classifier
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# processing zip
InPath = "small_img.zip"
OutPath = "images"


# Unzip and get names and path's to extracted images


def generate_name_path_dict(InPath=InPath, OutPath=OutPath):
    namePath = {}
    with ZipFile(InPath, "r") as zipObj:
        zipObj.extractall(OutPath)
        names = zipObj.namelist()
        paths = [OutPath + "/" + name for name in names]
        for i in range(len(names)):
            namePath.update({names[i]: paths[i]})
    return namePath


name_path = generate_name_path_dict()


# extract Text from pics
def preparation_to_OCR(items=name_path):
    objects = {key: Image.open(items[key]).convert("1") for key in items.keys()}

    return objects


ocr = preparation_to_OCR()


def get_text_from_page(items=ocr):
    parsed_pages = {key: pytesseract.image_to_string(items[key]).replace('\n', '') for key in items.keys()}

    return parsed_pages


text = get_text_from_page()


def match_word_and_faces(word, texts=text):
    result = []
    for k in texts.keys():
        if word in texts[k]:
            result.append(k)

    return result


# extract faces
def preparation_to_cv(keys, items=name_path):
    images = {k: cv.cvtColor(cv.imread(items[k]), cv.COLOR_BGR2GRAY) for k in keys}

    return images


def get_faces_from_page(items):
    faces = {k: face_cascade.detectMultiScale(items[k],
                                                scaleFactor=1.30,
                                                minNeighbors=5,
                                                minSize=(50, 50)) for k in items.keys()}

    # was used to test image recognition, no longer neded
    # def show_rects(items=generateNamePathDict()):
    #     headsFromImage = {}
    #     for key in items.keys():
    #         with Image.open(items[key]).convert("RGB") as pil_img:
    #             drawing = ImageDraw.Draw(pil_img)
    #             for x, y, w, h in faces[key]:
    #                 drawing.rectangle((x, y, x + w, y + h), outline="red")
    #                 headsFromImage.update({key: pil_img})
    #     return headsFromImage
    return faces


def crop_faces(keys, faces, items=name_path):
    faces_from_image = {}
    for key in keys:
        with Image.open(items[key]).convert("RGB") as pil_img:
            listFaces = [pil_img.crop((x, y, x + w, y + h)) for x, y, w, h in faces[key]]
        faces_from_image[key] = listFaces
    return faces_from_image


def contact_sheet_results(items):
    sheets = {}
    for k in items.keys():
        sheets[k] = Image.new(items[k][0].mode, (550, 110 * int(np.ceil(len(items[k]) / 5))))
        x = 0
        y = 0
        for face in items[k]:
            face.thumbnail((110, 110))
            sheets[k].paste(face, (x, y))
            if x + 110 == sheets[k].width:
                x = 0
                y = y + 110
            else:
                x = x + 110
    return sheets


def manager_func(word):
    # preparation_to_OCR()
    # get_text_from_page()
    match = match_word_and_faces(word=word)
    prepar = preparation_to_cv(keys=match)
    heads = get_faces_from_page(prepar)
    crop_face = crop_faces(keys=prepar, faces=heads)
    result = contact_sheet_results(crop_face)
    return result


if __name__ == '__main__':

    res = manager_func("a")
    for k,v in res.items():
        print(f"Resulf for page {k}")
        v.show()
