import tensorflow as tf
from flask import Flask, request,render_template
import cv2 
import os
from tensorflow import keras
import numpy as np
import meshplot as mp
from skimage.measure import marching_cubes_lewiner
from random import randint
from FLASK_1.utils import SAVE_PREDICTED_IMAGE_FROM_COLUME_DIRECTO,PREPARE_PREDICT_UNET,REMOVE_ALL_BALCK_IMAGE_MODI,SAVE_CROPPED_IMAGE,predictVolume_path_folders, save_img_from_unet_prediction,test_prep,size_detection_one_slide_VOLUME,size_detection_one_slide,predictVolume_path_folders_cropped




LABLE=["NOT_TUMOR","TUMOR"]


app=Flask(__name__)
@app.route("/")
def home ():
    return(render_template("home.html"))

@app.route("/upload.html",methods=["GET","POST"])
def upload():
    if request.method=="POST":
        IMG=request.files["img"]
        path=os.path.join("static/uploads",IMG.filename)
        IMG.save(path)
        model=tf.keras.models.load_model("models/MODEL_V2_acc_97")
        #read the img as np
        IMG_ARR=cv2.imread(path)
        #prepare img for model
        new_img=test_prep(IMG_ARR,260)

        pred=int(model.predict(new_img)[0][0])
        
        red = [255,0,0]
        blue=[0,0,255]
        if pred == 0:
            BOR_IMG=cv2.cv2.copyMakeBorder(IMG_ARR,20,20,20,20,cv2.BORDER_CONSTANT,value=tuple(red))
            path_modi=os.path.join("static/upload_modi",IMG.filename)
            cv2.imwrite(path_modi,BOR_IMG)
        else:
            BOR_IMG=cv2.cv2.copyMakeBorder(IMG_ARR,20,20,20,20,cv2.BORDER_CONSTANT,value=tuple(blue))
            path_modi=os.path.join("static/upload_modi",IMG.filename)
            cv2.imwrite(path_modi,BOR_IMG)
           

        return (render_template("upload.html",text=LABLE[pred],url=path_modi))
        
    return(render_template("upload.html"))

@app.route("/TEST.html",methods=["GET","POST"])
def VOLUME ():
     if request.method=="POST":
         UP = request.form.get("UPLOAD")
         if UP=='ENTER':
             slice=request.form.get("SLICE")
             if slice=='X':
                 #CHECK IF FOLDER IS CREATED OR NOT
                 if len(os.listdir("static/VOLUME")) == 0:
                     NAME='USER_0'
                     os.makedirs(f"static/VOLUME/{NAME}")
                     os.makedirs(f"static/VOLUME/{NAME}/X_SLICES")
                     os.makedirs(f"static/VOLUME/{NAME}/Y_SLICES")
                     os.makedirs(f"static/VOLUME/{NAME}/Z_SLICES")
                     os.makedirs(f"static/VOLUME/{NAME}/X_SLICES_PRED")
                     os.makedirs(f"static/VOLUME/{NAME}/Y_SLICES_PRED")
                     os.makedirs(f"static/VOLUME/{NAME}/Z_SLICES_PRED")
                     os.makedirs(f"static/VOLUME/{NAME}/X_SLICES_PRED/MASK")
                     os.makedirs(f"static/VOLUME/{NAME}/Y_SLICES_PRED/MASK")
                     os.makedirs(f"static/VOLUME/{NAME}/Z_SLICES_PRED/MASK")
                     os.makedirs(f"static/VOLUME/{NAME}/X_SLICES_PRED/SIZE")
                     os.makedirs(f"static/VOLUME/{NAME}/Y_SLICES_PRED/SIZE")
                     os.makedirs(f"static/VOLUME/{NAME}/Z_SLICES_PRED/SIZE")
                     os.makedirs(f"static/VOLUME/{NAME}/X_SLICES_PRED/CROP")
                     os.makedirs(f"static/VOLUME/{NAME}/Y_SLICES_PRED/CROP")
                     os.makedirs(f"static/VOLUME/{NAME}/Z_SLICES_PRED/CROP")
                 else:
                     NAME=f'USER_{len(os.listdir("static/VOLUME"))}'
                     os.makedirs(f'static/VOLUME/{NAME}')
                     os.makedirs(f"static/VOLUME/{NAME}/X_SLICES")
                     os.makedirs(f"static/VOLUME/{NAME}/Y_SLICES")
                     os.makedirs(f"static/VOLUME/{NAME}/Z_SLICES")
                     os.makedirs(f"static/VOLUME/{NAME}/X_SLICES_PRED")
                     os.makedirs(f"static/VOLUME/{NAME}/Y_SLICES_PRED")
                     os.makedirs(f"static/VOLUME/{NAME}/Z_SLICES_PRED")
                     os.makedirs(f"static/VOLUME/{NAME}/X_SLICES_PRED/MASK")
                     os.makedirs(f"static/VOLUME/{NAME}/Y_SLICES_PRED/MASK")
                     os.makedirs(f"static/VOLUME/{NAME}/Z_SLICES_PRED/MASK")
                     os.makedirs(f"static/VOLUME/{NAME}/X_SLICES_PRED/SIZE")
                     os.makedirs(f"static/VOLUME/{NAME}/Y_SLICES_PRED/SIZE")
                     os.makedirs(f"static/VOLUME/{NAME}/Z_SLICES_PRED/SIZE")
                     os.makedirs(f"static/VOLUME/{NAME}/X_SLICES_PRED/CROP")
                     os.makedirs(f"static/VOLUME/{NAME}/Y_SLICES_PRED/CROP")
                     os.makedirs(f"static/VOLUME/{NAME}/Z_SLICES_PRED/CROP")
                #GET THE LIST OF UPLOADED IMAGES
                 IMG=request.files.getlist("FILE")

                 for img in IMG:
                     UPLOAD_FOLDER = f'static/VOLUME/{NAME}/X_SLICES'
                     app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
                     img.save(os.path.join(app.config['UPLOAD_FOLDER'],img.filename))
                 return (render_template("TEST.html", text=f'{slice} SLICES UPLOADED SUCCESSFULLY'))

             NAME = f'USER_{len(os.listdir("static/VOLUME"))-1}'

             if slice == 'Y':
                 IMG = request.files.getlist("FILE")
                 for img in IMG:
                     UPLOAD_FOLDER = f'static/VOLUME/{NAME}/Y_SLICES'
                     app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
                     img.save(os.path.join(app.config['UPLOAD_FOLDER'], img.filename))
                 return (render_template("TEST.html", text=f'{slice} SLICES UPLOADED SUCCESSFULLY'))

             if slice == 'Z':

                 IMG = request.files.getlist("FILE")
                 for img in IMG:
                     UPLOAD_FOLDER = f'static/VOLUME/{NAME}/Z_SLICES'
                     app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
                     img.save(os.path.join(app.config['UPLOAD_FOLDER'], img.filename))

                 return (render_template("TEST.html", text=f'{slice} SLICES UPLOADED SUCCESSFULLY'))


         PRED=request.form.get("prd")
         X_DPI=request.form.get("X_DPI")
         Y_DPI=request.form.get("Y_DPI")
         Z_DPI=request.form.get("Z_DPI")
         X_TH=int(request.form.get("X_TH"))
         Y_TH =int(request.form.get("Y_TH"))
         Z_TH =int(request.form.get("Z_TH"))
         NAME = f'USER_{len(os.listdir("static/VOLUME")) - 1}'

         if PRED=='PREDICT':

             if len(os.listdir(f'static/VOLUME/{NAME}/Y_SLICES')) ==0 :
                 TEXT='PLS UPLOAD THE FULL SLICES'
                 return (render_template("TEST.html", text=TEXT))

             elif len(os.listdir(f'static/VOLUME/{NAME}/Z_SLICES')) ==0 :
                 TEXT='PLS UPLOAD THE FULL SLICES'
                 return (render_template("TEST.html", text=TEXT))
             else:
                 TEXT = 'THE PREDICTION IS READY'
                 X_DIR=f'static/VOLUME/{NAME}/X_SLICES'
                 Y_DIR= f'static/VOLUME/{NAME}/Y_SLICES'
                 Z_DIR=f'static/VOLUME/{NAME}/Z_SLICES'
                 #xl, yl, zl = REMOVE_ALL_BALCK_IMAGE(X_DIR, Y_DIR, Z_DIR)
                 xl, yl, zl = REMOVE_ALL_BALCK_IMAGE_MODI(X_DIR, Y_DIR, Z_DIR)
                 model=keras.models.load_model("models/U_NET_4_LEVEL_400_EPOCHS_867_WITHOUT_AUG_TRAIN_IMG_and_with_99_img")

                 out=predictVolume_path_folders(xl=xl,yl=yl,zl=zl,X_DIR=X_DIR,Y_DIR=Y_DIR,Z_DIR=Z_DIR,model=model)
                 vertices, faces, _, _ = marching_cubes_lewiner(out)
                 mp.offline()
                 p = mp.plot(vertices, faces, return_plot=True)
                 saved_meshplot_path=f"static/VOLUME/{NAME}/{randint(1,100000)}.html"
                 p.save(saved_meshplot_path)
                 X_DIR_MASK=f"static/VOLUME/{NAME}/X_SLICES_PRED/MASK"
                 Y_DIR_MASK=f"static/VOLUME/{NAME}/Y_SLICES_PRED/MASK"
                 Z_DIR_MASK=f"static/VOLUME/{NAME}/Z_SLICES_PRED/MASK"
                 X_LIST_PATH=sorted(xl, key=len)
                 Y_LIST_PATH=sorted(yl, key=len)
                 Z_LIST_PATH=sorted(zl, key=len)

                 SAVE_PREDICTED_IMAGE_FROM_COLUME_DIRECTO(out=out,X_DIR=X_DIR_MASK,Y_DIR=Y_DIR_MASK,Z_DIR=Z_DIR_MASK,LIST_NAME_X=X_LIST_PATH,LIST_NAME_Y=Y_LIST_PATH,LIST_NAME_Z=Z_LIST_PATH    )


                 #PREDIT SIZE AND MASK AND SAVE THE MASK AND IMAGE WITH SIZE
                 SIZE_X=[]
                 SIZE_Z=[]
                 SIZE_Y=[]

                 try:
                     for name_x in sorted(os.listdir(X_DIR_MASK),key=len):
                         SAVED_IMAGE_PATH = f"static/VOLUME/{NAME}/X_SLICES_PRED/MASK/{name_x}"
                         if np.count_nonzero(cv2.imread(SAVED_IMAGE_PATH)) == 0:
                             pass

                         else:
                             SIZE_x = size_detection_one_slide_VOLUME(SAVED_IMAGE_PATH, X_DIR, name_x, f"static/VOLUME/{NAME}/X_SLICES_PRED/SIZE",DPI=int(X_DPI))
                             if SIZE_x==None:
                                 SIZE_x=0
                                 SIZE_X.append(SIZE_x)
                             else:
                                 SIZE_X.append(SIZE_x)
                 except:
                     pass
                 try:
                     for name_y in sorted(os.listdir(Y_DIR_MASK),key=len):
                         SAVED_IMAGE_PATH = f"static/VOLUME/{NAME}/Y_SLICES_PRED/MASK/{name_y}"

                         if np.count_nonzero(cv2.imread(SAVED_IMAGE_PATH)) == 0:
                             pass

                         else:
                             SIZE_y = size_detection_one_slide_VOLUME(SAVED_IMAGE_PATH, Y_DIR, name_y, f"static/VOLUME/{NAME}/Y_SLICES_PRED/SIZE",DPI=int(Y_DPI))
                             if SIZE_y==None:
                                 SIZE_y=0
                                 SIZE_Y.append(SIZE_y)
                             else:
                                 SIZE_Y.append(SIZE_y)
                 except:
                     pass
                 try:
                     for name_z in sorted(os.listdir(Z_DIR_MASK),key=len):
                         SAVED_IMAGE_PATH = f"static/VOLUME/{NAME}/Z_SLICES_PRED/MASK/{name_z}"
                         if np.count_nonzero(cv2.imread(SAVED_IMAGE_PATH)) == 0:
                             pass

                         else:
                             SIZE_z = size_detection_one_slide_VOLUME(SAVED_IMAGE_PATH, Z_DIR, name_z, f"static/VOLUME/{NAME}/Z_SLICES_PRED/SIZE",DPI=int(Z_DPI))
                             if SIZE_z == None:
                                 SIZE_z = 0
                                 SIZE_Z.append(SIZE_z)
                             else:
                                 SIZE_Z.append(SIZE_z)
                 except:
                     pass
                 X_DIR_CROP=f"static/VOLUME/{NAME}/X_SLICES_PRED/CROP"
                 Y_DIR_CROP = f"static/VOLUME/{NAME}/Y_SLICES_PRED/CROP"
                 Z_DIR_CROP = f"static/VOLUME/{NAME}/Z_SLICES_PRED/CROP"
                 SAVE_CROPPED_IMAGE(X_DIR,Y_DIR,Z_DIR,X_DIR_MASK,Y_DIR_MASK,Z_DIR_MASK,X_LIST_PATH,Y_LIST_PATH,Z_LIST_PATH,X_DIR_CROP,Y_DIR_CROP,Z_DIR_CROP)

                 X_IMG_PATH=[f"static/VOLUME/{NAME}/X_SLICES_PRED/SIZE/{IMG_NAME}" for IMG_NAME in os.listdir(f"static/VOLUME/{NAME}/X_SLICES_PRED/SIZE") ]
                 Y_IMG_PATH=[f"static/VOLUME/{NAME}/Y_SLICES_PRED/SIZE/{IMG_NAME}" for IMG_NAME in os.listdir(f"static/VOLUME/{NAME}/Y_SLICES_PRED/SIZE") ]
                 Z_IMG_PATH=[f"static/VOLUME/{NAME}/Z_SLICES_PRED/SIZE/{IMG_NAME}" for IMG_NAME in os.listdir(f"static/VOLUME/{NAME}/Z_SLICES_PRED/SIZE") ]


                 xl_cr, yl_cr, zl_cr = REMOVE_ALL_BALCK_IMAGE_MODI(X_DIR_CROP, Y_DIR_CROP, Z_DIR_CROP)
                 out_cr=predictVolume_path_folders_cropped(xl_cr,yl_cr,zl_cr,X_DIR_CROP,Y_DIR_CROP,Z_DIR_CROP)
                 vertices, faces, _, val_cr = marching_cubes_lewiner(out_cr)
                 mp.offline()
                 shading = {"flat": False, "colormap": "gray"}
                 p = mp.plot(vertices,faces, c=val_cr,shading=shading,return_plot=True)
                 saved_meshplot_path_crop = f"static/VOLUME/{NAME}/CROP_{randint(1, 1000000)}.html"
                 p.save(saved_meshplot_path_crop)

                 X_TH_CM_2=(X_TH/10)*(X_TH/10)
                 Y_TH_CM_2 = (Y_TH / 10) * (Y_TH / 10)
                 Z_TH_CM_2 = (Z_TH / 10) * (Z_TH / 10)
                 V_X=np.sum(np.asarray(SIZE_X)+X_TH_CM_2)
                 V_Y = np.sum(np.asarray(SIZE_Y)+Y_TH_CM_2)
                 V_Z=np.sum(np.asarray(SIZE_Z)+Z_TH_CM_2)

                 VOLU=int((V_Y+V_X+V_Z)/3)
                 return (render_template("TEST.html", text=TEXT,ZERO_TEXT=f'REMOVING ALL ZEROS IS DONE {len(xl)}--{len(yl)}---{len(zl)}',MESH_PATH=saved_meshplot_path,X_PATH=X_IMG_PATH,Y_PATH=Y_IMG_PATH,Z_PATH=Z_IMG_PATH,CROP_MESH_PATH=saved_meshplot_path_crop,VOLUME=VOLU,VOLUME_X=V_X,VOLUME_Y=V_Y,VOLUME_Z=V_Z))

     else:
         return(render_template("TEST.html"))


@app.route("/SIZE_DETE.html",methods=["GET","POST"])
def size_dete():
    if request.method == "POST":
        IMG = request.files.getlist("img")
        NAME = f'USER_{len(os.listdir("static/SIZE_DETE"))}'
        os.makedirs(f'static/SIZE_DETE/{NAME}')
        os.makedirs(f'static/SIZE_DETE/{NAME}/ORG')
        os.makedirs(f'static/SIZE_DETE/{NAME}/PRED')
        os.makedirs(f'static/SIZE_DETE/{NAME}/SIZE_IMG')
        for img in IMG:
            UPLOAD_FOLDER = f'static/SIZE_DETE/{NAME}/ORG'
            app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
            img.save(os.path.join(app.config['UPLOAD_FOLDER'], img.filename))
        try:

            for name in os.listdir(f"static/SIZE_DETE/{NAME}/ORG"):
                ORG_PFOLDER_PATH =f"static/SIZE_DETE/{NAME}/ORG"
                IMG_PATH = f"static/SIZE_DETE/{NAME}/ORG/{name}"
                model = keras.models.load_model("models/U_NET_4_LEVEL_400_EPOCHS_867_WITHOUT_AUG_TRAIN_IMG_and_with_99_img")

                PRED = PREPARE_PREDICT_UNET(IMG_PATH=IMG_PATH, model=model)
                save_img_from_unet_prediction(PRED, f"static/SIZE_DETE/{NAME}/PRED", name)

                SAVED_IMAGE_PATH = f"static/SIZE_DETE/{NAME}/PRED/{name}"
                SIZE = size_detection_one_slide(SAVED_IMAGE_PATH, ORG_PFOLDER_PATH, name,SAVE_FOLDER_DEST=f"static/SIZE_DETE/{NAME}/SIZE_IMG")
                IMG_PATH = [f"static/SIZE_DETE/{NAME}/SIZE_IMG/{IMG_NAME}" for IMG_NAME in os.listdir(f"static/SIZE_DETE/{NAME}/SIZE_IMG")]

            return (render_template('SIZE_DETE.html',TEXT="UPLOADS IS DONE",PATH=IMG_PATH))
        except:
            pass
    else:
        return (render_template('SIZE_DETE.html'))

@app.route("/home.html")
def home_1 ():
    return(render_template("home.html"))




if __name__=='__main__':
    app.run(debug=True)