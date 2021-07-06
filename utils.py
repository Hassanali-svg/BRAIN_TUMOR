import tensorflow as tf
from flask import Flask, redirect, url_for, request,render_template
import cv2
import os
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import meshplot as mp
from skimage.measure import marching_cubes_lewiner
import imutils
from imutils import perspective
from imutils import contours
from random import randint

def SAVE_PREDICTED_IMAGE_FROM_COLUME_DIRECTO(out,X_DIR,Y_DIR,Z_DIR,LIST_NAME_X,LIST_NAME_Y,LIST_NAME_Z):
    SLICE_X = True
    SLICE_Y = True
    SLICE_Z = True

    c_x=0
    c_y=0
    c_z=0

    if SLICE_X:
        for name in LIST_NAME_X:
            plt.imsave(os.path.join(X_DIR, f"{name}"), out[c_x, :, :], cmap='gray')
            c_x = c_x + 1

    if SLICE_Y:

        for name in LIST_NAME_Y:
            plt.imsave(os.path.join(Y_DIR, f"{name}"), out[:, c_y, :], cmap='gray')
            c_y = c_y + 1

    if SLICE_Z:
        for name in LIST_NAME_Z:
            plt.imsave(os.path.join(Z_DIR, f"{name}"), out[:, :, c_z], cmap='gray')

            c_z = c_z + 1


def test_prep(IMG_ARR,IMG_SIZE):
  ''' this func resize the test image and read it in gray scale'''
  img_arr=cv2.cvtColor(IMG_ARR, cv2.COLOR_BGR2GRAY)
  new=cv2.resize(img_arr,(IMG_SIZE,IMG_SIZE))
  return new.reshape(-1,IMG_SIZE,IMG_SIZE,1)


def REMOVE_ALL_BALCK_IMAGE(X_DIR, Y_DIR, Z_DIR):
    xl = []
    yl = []
    zl = []
    for image_x in os.listdir(X_DIR):
        img = cv2.imread(os.path.join(X_DIR, f"{image_x}"))
        if np.count_nonzero(img) != 0:
            xl.append(image_x)

    for image_y in os.listdir(Y_DIR):
        img = cv2.imread(os.path.join(Y_DIR, f"{image_y}"))
        if np.count_nonzero(img) != 0:
            yl.append(image_y)

    for image_z in os.listdir(Z_DIR):
        img = cv2.imread(os.path.join(Z_DIR, f"{image_z}"))
        if np.count_nonzero(img) != 0:
            zl.append(image_z)

    return xl, yl, zl

def REMOVE_ALL_BALCK_IMAGE_MODI(X_DIR, Y_DIR, Z_DIR):
    xl = []
    yl = []
    zl = []
    for image_x in os.listdir(X_DIR):
        xl.append(image_x)

    for image_y in os.listdir(Y_DIR):
        yl.append(image_y)

    for image_z in os.listdir(Z_DIR):

        zl.append(image_z)

    return sorted(xl, key=len), sorted(yl, key=len), sorted(zl, key=len)
def scale_Img(img, height, width):
    return cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_LINEAR)


def predictVolume_path_folders(xl, yl, zl,model,X_DIR,Y_DIR,Z_DIR,toBin=True):
    xMax = len(xl)
    yMax = len(yl)
    zMax = len(zl)
    xl = xl
    yl = yl
    zl = zl
    SLICE_X = True
    SLICE_Y = True
    SLICE_Z = True
    outImgX = np.zeros((xMax, yMax, zMax))
    outImgY = np.zeros((xMax, yMax, zMax))
    outImgZ = np.zeros((xMax, yMax, zMax))

    cnt = 0.0
    if SLICE_X:
        cnt += 1.0
        for i in range(xMax):
            path_x = os.path.join(X_DIR, f"{sorted(xl, key=len)[i]}")
            IMG_TEST = cv2.cvtColor(plt.imread(path_x), cv2.COLOR_BGR2GRAY)
            #IMG_TEST = plt.imread(path_x,format='gray')

            img = scale_Img(IMG_TEST, 200, 200)[np.newaxis, :, :, np.newaxis]
            tmp = model.predict(img)[0, :, :, 0]
            outImgX[i, :, :] = scale_Img(tmp, yMax, zMax)

    if SLICE_Y:

        cnt += 1.0
        for i in range(yMax):
            path_y = os.path.join(Y_DIR, f"{sorted(yl, key=len)[i]}")
            IMG_TEST = cv2.cvtColor(plt.imread(path_y), cv2.COLOR_BGR2GRAY)
            #IMG_TEST = plt.imread(path_y,format='gray')

            img = scale_Img(IMG_TEST, 200, 200)[np.newaxis, :, :, np.newaxis]
            tmp = model.predict(img)[0, :, :, 0]
            outImgY[:, i, :] = scale_Img(tmp, xMax, zMax)

    if SLICE_Z:
        cnt += 1.0
        for i in range(zMax):
            path_z = os.path.join(Z_DIR, f"{sorted(zl, key=len)[i]}")
            IMG_TEST = cv2.cvtColor(plt.imread(path_z), cv2.COLOR_BGR2GRAY)
            #IMG_TEST = plt.imread(path_z)

            img = scale_Img(IMG_TEST, 200, 200)[np.newaxis, :, :, np.newaxis]
            tmp = model.predict(img)[0, :, :, 0]
            outImgZ[:, :, i] = scale_Img(tmp, xMax, yMax)

    outImg = (outImgX + outImgY + outImgZ) / cnt
    if (toBin):
        outImg[outImg > 0.5] = 1.0
        outImg[outImg <= 0.5] = 0.0
    return outImg


def PREPARE_PREDICT_UNET(IMG_PATH,model,MODEL_IMG_SIZE=(200,200)):
    IMG_TEST = cv2.cvtColor(plt.imread(IMG_PATH), cv2.COLOR_BGR2GRAY)

    # RESIZE ORG IMAGE TO (200,200) WITH SACLING /255
    R_IMG_TEST = resize_img(MODEL_IMG_SIZE, IMG_TEST) /255

    # MODEL
    PRED_MASK = model.predict(R_IMG_TEST.reshape(-1, 200, 200, 1))
    RESH = PRED_MASK.reshape(200, 200)
    PRED = resize_img((IMG_TEST.shape[1], IMG_TEST.shape[0]), RESH)
    PRED[PRED < 0.5] = 0
    PRED[PRED > 0.5] = 1
    return PRED


def resize_img(IMG_SIZE, IMG):

    '''RESIZE FUNCTION'''
    return cv2.resize(IMG, IMG_SIZE, interpolation=cv2.INTER_LINEAR)

'''THIS FUNCTION PREPARE THE INPUT IMAGE FOR UNET MODEL THEN THE MODEL PREDICT IT 
    ADN THIS FUCNTION REURN THR PREDICTED MASK WITH THE SAME DIM OF THE ORGINAL INPUT IMAGE'''

def save_img_from_unet_prediction(PRED_NP_ARRAY,dest_path,img_name):
    '''THIS FUNCTIO WILL TAKE predicted iamge from unet ,and the dest folder name ,the name of image to save in the folder
    return nothing but the folder will be conatain the saved image'''

    img = cv2.convertScaleAbs(PRED_NP_ARRAY, alpha=(255.0))
    cv2.imwrite(f"{dest_path}/{img_name}", img)


def size_detection_one_slide_VOLUME(SAVED_PRED_IMG_PATH, ORGINAL_IMAGE_PATH, IMAGE_NAME,SIZE_IMAGE_PATH,DPI):
    if SAVED_PRED_IMG_PATH==None:
        pass
    else:
        MASK = cv2.imread(f"{SAVED_PRED_IMG_PATH}")
        org_A = cv2.imread(f"{ORGINAL_IMAGE_PATH}/{IMAGE_NAME}")

        RESIZIED_MASK=resize_img((org_A.shape[1], org_A.shape[0]), MASK)
        gray = cv2.cvtColor(RESIZIED_MASK, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        edge = cv2.Canny(gray, 10, 100)
        edge = cv2.dilate(edge, None, iterations=1)
        edge = cv2.erode(edge, None, iterations=1)

        # find contours
        cont= cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cont = imutils.grab_contours(cont)

        (cont, _) = contours.sort_contours(cont)

        #M = cv2.getRotationMatrix2D((org.shape[1] / 2, org.shape[0] / 2), 90, 1)
        #org = cv2.warpAffine(org, M, (org.shape[1], org.shape[0]))
        for c in cont:
            if cv2.contourArea(c) < 50:
                continue
            org = org_A.copy()
            cv2.drawContours(org, [c], -1, (0, 0, 255), 4)

            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)
            # to be in  format tl,tr,br,bl
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            #print(box)

            NO_OF_NOE_ZERO=np.count_nonzero(MASK)
            SIZE=(NO_OF_NOE_ZERO/(DPI*DPI))*2.54
            final=cv2.putText(org, "{:.4f}cm^2".format(SIZE), tuple(box[0].astype("int")), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imwrite(f"{SIZE_IMAGE_PATH}/{IMAGE_NAME}",final)


            #cv2.imshow("SOKA", org)
            #cv2.waitKey(100)
            return SIZE

def size_detection_one_slide(SAVED_PRED_IMG_PATH, ORGINAL_IMAGE_PATH, IMAGE_NAME,SAVE_FOLDER_DEST):
    if SAVED_PRED_IMG_PATH==None:
        pass
    else:
        image = cv2.imread(f"{SAVED_PRED_IMG_PATH}")


        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        edge = cv2.Canny(gray, 10, 100)
        edge = cv2.dilate(edge, None, iterations=1)
        edge = cv2.erode(edge, None, iterations=1)

        # find contours
        cont = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cont = imutils.grab_contours(cont)

        (cont, _) = contours.sort_contours(cont)


        #org = cv2.imread(f"{ORGINAL_IMAGE_PATH}/{IMAGE_NAME}")
        org = cv2.imread(f"{ORGINAL_IMAGE_PATH}/{IMAGE_NAME}")
        #M = cv2.getRotationMatrix2D((org.shape[1] / 2, org.shape[0] / 2), 90, 1)
        #org = cv2.warpAffine(org, M, (org.shape[1], org.shape[0]))
        for c in cont:
            if cv2.contourArea(c) < 150:
                continue
            org = org.copy()
            cv2.drawContours(org, [c], -1, (0, 0, 255), 4)

            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)
            # to be in  format tl,tr,br,bl
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            #print(box)

            NO_OF_NOE_ZERO=np.count_nonzero(image)
            SIZE=(NO_OF_NOE_ZERO/(96*96))*2.54
            final=cv2.putText(org, "{:.4f}cm^2".format(SIZE), tuple(box[0].astype("int")), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imwrite(f"{SAVE_FOLDER_DEST}/{IMAGE_NAME}",final)
            #cv2.imshow("SOKA", org)
            #cv2.waitKey(100)
            return SIZE

def SAVE_CROPPED_IMAGE(X_ORG_PATH,Y_ORG_PATH,Z_ORG_PATH,X_MASK_PATH,Y_MASK_PATH,Z_MASK_PATH,xl,yl,zl,X_CROP_PATH,Y_CROP_PATH,Z_CROP_PATH):
    for x_name in sorted(xl, key=len):
        # croped image
        org_x=cv2.imread(f"{X_ORG_PATH}/{x_name}")
        mask_x=cv2.imread(f"{X_MASK_PATH}/{x_name}")
        mask_out = cv2.subtract(mask_x, org_x)
        mask_out = cv2.subtract(mask_x, mask_out)
        cv2.imwrite(f"{X_CROP_PATH}/{x_name}", mask_out)

    for y_name in sorted(yl, key=len):
        # croped image
        org_y = cv2.imread(f"{Y_ORG_PATH}/{y_name}")
        mask_y = cv2.imread(f"{Y_MASK_PATH}/{y_name}")
        mask_out = cv2.subtract(mask_y, org_y)
        mask_out = cv2.subtract(mask_y, mask_out)
        cv2.imwrite(f"{Y_CROP_PATH}/{y_name}", mask_out)

    for z_name in sorted(zl, key=len):
        # croped image
        org_z = cv2.imread(f"{Z_ORG_PATH}/{z_name}")
        mask_z = cv2.imread(f"{Z_MASK_PATH}/{z_name}")
        mask_out = cv2.subtract(mask_z, org_z)
        mask_out = cv2.subtract(mask_z, mask_out)
        cv2.imwrite(f"{Z_CROP_PATH}/{z_name}", mask_out)


def predictVolume_path_folders_cropped(xl, yl, zl,X_DIR,Y_DIR,Z_DIR):
    SLICE_X = True
    SLICE_Y = True
    SLICE_Z = True
    xMax = len(xl)
    yMax = len(yl)
    zMax = len(zl)
    xl = xl
    yl = yl
    zl = zl

    outImgX = np.zeros((xMax, yMax, zMax))
    outImgY = np.zeros((xMax, yMax, zMax))
    outImgZ = np.zeros((xMax, yMax, zMax))

    cnt = 0.0
    if SLICE_X:
        cnt += 1.0
        for i in range(len(xl)):
            path_x = os.path.join(X_DIR, f"{sorted(xl, key=len)[i]}")
            IMG_TEST = cv2.cvtColor(plt.imread(path_x), cv2.COLOR_BGR2GRAY)
            outImgX[i, :, :] = IMG_TEST

    if SLICE_Y:

        cnt += 1.0
        for i in range(len(yl)):
            path_x = os.path.join(Y_DIR, f"{sorted(yl, key=len)[i]}")
            IMG_TEST = cv2.cvtColor(plt.imread(path_x), cv2.COLOR_BGR2GRAY)
            outImgY[:, i, :] = IMG_TEST

    if SLICE_Z:

        cnt += 1.0
        for i in range(len(zl)):
            path_x = os.path.join(Z_DIR, f"{sorted(zl, key=len)[i]}")
            IMG_TEST = cv2.cvtColor(plt.imread(path_x), cv2.COLOR_BGR2GRAY)
            outImgZ[:, :, i] = IMG_TEST

    outImg = (outImgX + outImgY + outImgZ) / cnt

    return outImg