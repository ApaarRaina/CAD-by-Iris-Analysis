import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

def circle_intensity(img, x0, y0, r, num_points=360):
    theta = np.linspace(0, 2 * np.pi, num_points)
    x_circle = x0 + r * np.cos(theta)
    y_circle = y0 + r * np.sin(theta)
    values = map_coordinates(img, [y_circle, x_circle], order=1, mode='reflect')
    return values  # 1D array of intensity values along the circle


def Iris_localisation(img):

    img=cv.GaussianBlur(img,(9,9),sigmaX=2)

    img_w,img_h=img.shape[1],img.shape[0]
    radii=np.linspace(0.1*img_w,0.5*img_w,400)

    center_x=np.random.randint(img_w/8,7*img_w/8,200)
    center_y=np.random.randint(img_h/8,7*img_h/8,200)

    h,w=img.shape[0],img.shape[1]
    coord_x=np.indices((h,w))[1]
    coord_y=np.indices((h,w))[0]

    max_grad=[]
    r_max=[]
    x_max=[]
    y_max=[]

    for i in range(len(center_x)):
          x,y=center_x[i],center_y[i]
          r_intensity=[]
          for r in radii:
              circle=circle_intensity(img,x,y,r)
              intensity=np.mean(circle) #integration I(x,y) around the circle
              r_intensity.append(intensity)

          r_grad=np.gradient(r_intensity)  # d/dr
          max_grad.append(np.max(r_grad))
          r_max.append(np.argmax(r_grad))
          x_max.append(x)
          y_max.append(y)


    max_grad=np.array(max_grad)
    grad,index=np.max(max_grad),np.argmax(max_grad)  # max r,x,y
    best_radius=radii[r_max[index]]
    x_value=x_max[index]
    y_value=y_max[index]

    r_intensity=[]
    for r in radii:
        circle=circle_intensity(img,x,y,r)
        intensity=np.mean(circle) #integration I(x,y) around the circle
        r_intensity.append(intensity)

    r_grad=np.gradient(r_intensity)
    peak_indices, _ = find_peaks(r_grad)

    valid_peaks = []
    for i in peak_indices:
        r_candidate = radii[i]
        if abs(r_candidate - best_radius) >= 0.4 * best_radius:
            valid_peaks.append((r_candidate, r_grad[i]))  # (radius, gradient value)
    valid_peaks = sorted(valid_peaks, key=lambda x: -x[1])

    second_radius=valid_peaks[0][0]

    pupil_radius,iris_radius=min([best_radius,second_radius]),max([best_radius,second_radius])

    return iris_radius,pupil_radius,x_value,y_value