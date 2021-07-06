
'''
path=os.path.join("static/uploads","F1.png")
plt.imread(path)
plt.figure(figsize=[10,10],edgecolor='red',linewidth=10)
plt.text(40,-10, f"the prediction is", bbox=dict(facecolor='red', alpha=0.1),fontsize=15,color='red')

plt.show()
path_modi=os.path.join("static/upload_modi","new_1.png")

plt.savefig(path_modi)


import sys
import flask
import tensorflow as tf
import os
if 1==1:

    g=0



h=g
print(g)

import cv2

import numpy as np
MASK=cv2.imread("static/VOLUME/USER_5/Z_SLICES_PRED/MASK/Z_SLICE52.png")

ORG=cv2.imread("static/VOLUME/USER_5/Z_SLICES/Z_SLICE52.png")



#src1_mask=cv2.cvtColor(MASK,cv2.COLOR_GRAY2BGR)#change mask to a 3 channel image
mask_out=cv2.subtract(MASK,ORG)
mask_out=cv2.subtract(MASK,mask_out)

cv2.imshow('soak',mask_out)
cv2.waitKey(0)

org_x = cv2.imread(f"static/VOLUME/USER_11/X_SLICES/X_")
mask_x = cv2.imread(f"static/VOLUME/USER_11/X_SLICES_PRED/MASK/{x_name}")
print()
for x_name in sorted(os.listdir("static/VOLUME/USER_11/X_SLICES"), key=len):
    # croped image
    org_x = cv2.imread(f"static/VOLUME/USER_11/X_SLICES/{x_name}")
    #cv2.waitKey(100)
    mask_x = cv2.imread(f"static/VOLUME/USER_11/X_SLICES_PRED/MASK/{x_name}")
    mask_out = cv2.subtract(mask_x, org_x)
    mask_out = cv2.subtract(mask_x, mask_out)
    #cv2.imwrite(f"{X_CROP_PATH}/{x_name}", mask_out)
'''
import numpy as np
P=[1,2,3,4]

print(np.asarray(P)+1)

print((4/10)*(4/10))
