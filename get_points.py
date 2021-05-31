import cv2 as cv
import numpy as np


# MÉTODO PARA SACAR LOS PUNTOS DE LOS POLÍGONOS 
# TOMAR ALGO PARECIDO A UN RECTANGULO (QUE PUEDE ESTAR ROTADO)
# LOS PUNTOS SE ESCOGEN USANDO EL CLICK IZQUIERDO 

def click_event(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        pts.append([x, y])
        
        # Displaying the point
        cv.circle(img, (x, y), 10, color, 10)
        
        # Drawing lines
        if(len(pts) > 1 ):
            cv.line(img, (pts[-2][0], pts[-2][1]), (pts[-1][0], pts[-1][1]), color, 10)
        if(len(pts) == 4):
            cv.line(img, (pts[0][0], pts[0][1]), (pts[3][0], pts[3][1]), color, 10)
            rects.append(np.array(pts))         
            pts.clear()
            
        cv.imshow('image', img)
        
########----- CORRER TODA ESTA SECCIÓN (BEGIN) -----####### 
        
img = cv.imread('Ensasho.jpg', 1)       # LEER A IMAGEN    
w = img.shape[1];   h = img.shape[0];   # SACAR LAS DIMENSIONES
color = (255, 0, 0)                     # PARA PINTAR SOBRE LA IMAGEN (AZUL)
pts = [];   rects = []                  # EN RECTS QUEDAN GUARDADO EL OUTPUT
    
# AJUSTAR DIMENSIONES DEL GRAFICADOR
cv.namedWindow('image', cv.WINDOW_NORMAL); cv.resizeWindow('image', w, h)

#GRAFICAR Y LLAMAR A LA FUNCIÓN QUE CONVIERTE CLICKS EN PUNTOS
cv.imshow('image', img); cv.setMouseCallback('image', click_event)
cv.waitKey(0); cv.destroyAllWindows();

########----- CORRER TODA ESTA SECCIÓN (END) -----####### 


# LOS POLÍGONOS QUEDAN GUARDADOS COMO NP.ARRAYS EN rects
# 1) img_file:  NOMBRE DEL ARCHIVO DE LA IMAGEN (SI NO ESTÁ EN LA 
#               MISMA CARPETA HAY QUE PASAR TODO EL PATH COMPLETO)
# 2) save_name: NOMBRE DE LOS ARCHIVOS QUE SE VAN A CREAR A PARTIR
#               DE LOS RECORTES (NO INCLUIR .JPG AL FINAL!)
#               EL CÓDIGO AGREGA NÚMEROS AL FINAL PARA DIFERENCIAR
#               LOS ARCHIVOS (f1.jpg, f2.jpg, f3.jpg ...)


def crop(img_file, save_name, rects, idx):
    img = cv.imread(img_file)
    # points for test.jpg
    for pol in rects:
        # Get min_area_rect from polygon
        rect = cv.minAreaRect(pol)
        
        # Order of points: bottom L, top L, top R, bottom R
        box = cv.boxPoints(rect)
        box = np.int0(box)
    
        # get width and height of the detected rectangle
        width = int(rect[1][0]);    height = int(rect[1][1])
        
        src_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been
        # straightened
        dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0],
                            [width-1, height-1]], dtype="float32")
    
        # the perspective transformation matrix
        M = cv.getPerspectiveTransform(src_pts, dst_pts)
    
        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv.warpPerspective(img, M, (width, height))
        
        # rotate
        width = warped.shape[1];    height = warped.shape[0]
        if(width > height):
            warped = cv.rotate(warped, cv.ROTATE_90_CLOCKWISE)
        
        # AJUSTAR DIMENSIONES
        warped = cv.resize(warped, std_dims, interpolation = cv.INTER_AREA)
        
        # GUARDA
        save_path = save_name + str(idx) + ".jpg"
        cv.imwrite(save_path, warped); cv.waitKey(0)
        idx += 1
        # ROTADA
        rot_img = cv.rotate(warped, cv.ROTATE_180)
        save_path = save_name + str(idx) + ".jpg"
        cv.imwrite(save_path, rot_img); cv.waitKey(0)
        idx += 1
        #REFLEJADA
        ref_img = cv.flip(warped, 1)
        save_path = save_name + str(idx) + ".jpg"
        cv.imwrite(save_path, ref_img); cv.waitKey(0)
        idx += 1
        #ROTADA Y REFLEJADA
        ref_rot = cv.flip(rot_img, 1)
        save_path = save_name + str(idx) + ".jpg"
        cv.imwrite(save_path, ref_rot); cv.waitKey(0)
        idx += 1
        

###CORRER TODO ESTO###
idx = 1     ##CORRER SOLO UNA VEZ


std_dims = (64, 128) #W, H
img_file = "Ensasho.jpg"
save_name = "crop_img"
crop(img_file, save_name, rects, idx)