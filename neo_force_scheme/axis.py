import math

import numpy as np


# Read data from a csv


# original data set and max+min for each variable
# data_with_axis_points = np.empty(len(data) + len(data[0])*2)

def calculate_mean(data):
    # create an array with the size of the dimension of data
    mean = np.zeros(len(data[0]))

    # add the value of the variables in each point to the corresponding position in the array
    for n in range(len(data)):
        for m in range(len(data[0])):
            mean[m] = mean[m] + data[n][m]

    # divide each value by the length of the data
    for n in range(len(mean)):
        mean[n] = mean[n] / 3

    return mean


# data = raw dataset, index = which variable to be calculated
def find_min(data, index, mean):
    temp_min = np.zeros(len(data[0]))
    temp_min[index] = data[0][index]

    for n in range(len(data)):
        for m in range(len(data[0])):
            if m == index:
                if temp_min[m] > data[n][index]:
                    temp_min[m] = data[n][m]
            else:
                temp_min[m] = mean[m]

    return temp_min


def find_max(data, index, mean):
    temp_max = np.zeros(len(data[0]))
    temp_max[index] = data[0][index]

    for n in range(len(data)):
        for m in range(len(data[0])):
            if m == index:
                if temp_max[m] < data[n][index]:
                    temp_max[m] = data[n][m]
            else:
                temp_max[m] = mean[m]

    return temp_max


def oblique_projection(max_p, min_p, eye):

    #attempt 1
    difference = max_p - min_p
    # screenxp = x - y * cot(theta)
    # screenyp = z - y * cot(phi)

    #by default in plotly z is up, so actually z is 'y' that in a normal cordinate
    cot_screenx = eye.get('x') / eye.get('y')
    cot_screeny = eye.get('z') / eye.get('y')

    #print(eye.get('x'),' ', eye.get('y'))
    x_height = (difference[0] - difference[1] * cot_screenx)
    y_height = (difference[2] - difference[1] * cot_screeny)
    return x_height, y_height

    #TODO: find out how to calculate the viewport using up and eye variable in plotly.scence.camera.
    # I know eye is referring to the center of the viewport,
    # up means by default which axis is pointing up, but I don't understand how to combine two variables together
    # to calculate the screen coordinate
    """
    #attempt2
    cot_theta = eye.get('x') / eye.get('y')
    cot_phi = eye.get('z') / eye.get('y')

    H_matrix = np.array(
        [[1, 0, -(cot_theta), 0], [0, 1, -(cot_phi), 0], [0, 0, 1, 0],
         [0, 0, 0, 1]])
    M_matrix = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0, ], [0, 0, 0, 0], [0, 0, 0, 1]])

    print (distance * (M_matrix * H_matrix))
    

    #attempt3
    x_axis = np.array([1, 0])
    y_axis = np.array([0, 1])
    v1 = np.array([eye.get('x'), eye.get('z')])
    v2 = np.array([eye.get('z'), eye.get('y')])
    # yp = y - z * cot(phi)
    # xp = x - z * cot(theta)
    theta = np.arccos(v1.dot(x_axis) / (np.linalg.norm(v1) * np.linalg.norm(x_axis)))
    phi = np.arccos(v2.dot(y_axis) / (np.linalg.norm(v2) * np.linalg.norm(y_axis)))

    H_matrix = np.array(
        [[1, 0, -(np.cos(theta) / np.sin(theta)), 0], [0, 1, -(np.cos(phi) / np.sin(phi)), 0], [0, 0, 1, 0],
         [0, 0, 0, 1]])
    M_matrix = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0, ], [0, 0, 0, 0], [0, 0, 0, 1]])


    result = distance * ( M_matrix * H_matrix)
    return result
    
    
    
    """
"""
used to debug
min = np.array([0.0, 0.0, 0.0])
max = np.array([2.5, 0.0, 0.0])
dist = max - min
eye = np.array([2.5, 0, 0])
cos_theta = eye * /(np.linalg.norm(eye) * np.linalg.norm(dist))

point = np.array([2.5, 0.0, 0.0])
#oblique_projection(max, min, eye=dict(x=2.5, y=0.000001, z=0.000001))
#oblique_projection(max, min,eye=dict(x=0.000001, y=2.5, z=0.01))
#oblique_projection(max, min,eye=dict(x=0.000001, y=0.000001, z=2.5))

#normals = np.transpose(np.array([[2.5, 0, 0], [0, 0, 1]]))  # normals
normals = np.transpose(np.array([[0, 2.5, 0], [0, 0, 1]]))  # normals
#normals = np.transpose(np.array([[0, 0, 2.5], [0, 0, 1]]))  # normals
#point = np.array([0.1, 0.1, 0.1])  # point
coeff = np.linalg.lstsq(normals, point, rcond=None)[0]
print(coeff)
proj = point - np.dot(normals, coeff)
print(int(proj[0]), int(proj[1]), int(proj[2]))
"""