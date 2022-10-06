from shapely.geometry import Polygon 
import numpy as np

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[0,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def poly_area(x, y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def Cal_area_2poly(data1,data2):

    poly1 = Polygon(data1).convex_hull
    poly2 = Polygon(data2).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0
    else:
        inter_area = poly1.intersection(poly2).area 
    return inter_area

def _box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is positive Z
        corners2: numpy array (8,3), assume up direction is positive Z
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    todo (kent): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,1]) for i in [4,6,2,0]]
    rect2 = [(corners2[i,0], corners2[i,1]) for i in [4,6,2,0]] 
    
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    inter_area = Cal_area_2poly(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    
    ymax = min(corners1[1,2], corners2[1,2])
    ymin = max(corners1[0,2], corners2[0,2])

    inter_vol = inter_area * max(0.0, ymax-ymin)
    
    vol1 = area1 * (corners1[1,2] - corners1[0,2])
    vol2 = area2 * (corners2[1,2] - corners2[0,2])
    iou_3d = inter_vol / (vol1 + vol2 - inter_vol)
    return iou_3d

def _calculate_box_ious_3D(bboxes1, bboxes2):
    """ Compute multiple 3D bounding boxes' IoUs.
    Input:
        bboxes1 : numpy array (N * 24) 
        bboxes2 : numpy array (M * 24)
        each bbox : numpy array (24,), [x0, x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3, y4, y5, y6, y7, z0, z1, z2, z3, z4, z5, z6, z7]
    Output:
        ious : numpy array (N, M)
    """
    dim1 = bboxes1.shape[0]
    dim2 = bboxes2.shape[0]

    corners_1 = np.zeros((dim1, 8, 3))
    for i in range(dim1):
        temp = np.zeros((8,3))
        temp[:, 0] = bboxes1[i][:8]
        temp[:, 1] = bboxes1[i][8:16]
        temp[:, 2] = bboxes1[i][16:]
        corners_1[i] = temp

    corners_2 = np.zeros((dim2, 8, 3))
    for i in range(dim2):
        temp = np.zeros((8,3))
        temp[:, 0] = bboxes2[i][:8]
        temp[:, 1] = bboxes2[i][8:16]
        temp[:, 2] = bboxes2[i][16:]
        corners_2[i] = temp

    ious = np.zeros((dim1, dim2))
    for i in range(dim1):
        for j in range(dim2):
            iou = _box3d_iou(corners_1[i], corners_2[j])
            ious[i][j] = iou
    return ious

if __name__=='__main__':
    IoU_3D = _calculate_box_ious_3D(bboxes1, bboxes2)
