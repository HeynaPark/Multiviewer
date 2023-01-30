import cv2
import numpy as np
import os
import json
import natsort

# print(cv2.__version__)
# print(os.getcwd())

src = []
src_raw = []
src_w = 1920
src_h = 1080
cam_pts = []
world_pts = []

class Point:
    x = 0.0
    y = 0.0


# src import
src_path = '/home/hnpark/data/1/'
src_list = os.listdir(src_path)
src_list = natsort.natsorted(src_list)
src_num = len(src_list)


# world img import
world_img = cv2.imread('/home/hnpark/data/IceLink.png')


# calc viewer row & resize rate
multiview_row = 1
while 1:
    multiview_row += 1
    if multiview_row ** 2 > src_num:
        break
resize_rate = 1 / multiview_row


# json import (points)
json_file = open('/home/hnpark/data/1.pts')
json_data = json.load(json_file)
src_points = [[0 for j in range(5)] for i in range(src_num)]
for i in range(src_num):
    src_points[i][0] = json_data['points'][i]['dsc_id']
    x1 = json_data['points'][i]['pts_3d']['X1']
    x2 = json_data['points'][i]['pts_3d']['X2']
    x3 = json_data['points'][i]['pts_3d']['X3']
    x4 = json_data['points'][i]['pts_3d']['X4']
    y1 = json_data['points'][i]['pts_3d']['Y1']
    y2 = json_data['points'][i]['pts_3d']['Y2']
    y3 = json_data['points'][i]['pts_3d']['Y3']
    y4 = json_data['points'][i]['pts_3d']['Y4']
    src_points[i][1] = (x1, y1)
    src_points[i][2] = (x2, y2)
    src_points[i][3] = (x3, y3)
    src_points[i][4] = (x4, y4)
world_pts.append(json_data['stadium'])
wx1 = json_data['world_coords']['X1']
wx2 = json_data['world_coords']['X2']
wx3 = json_data['world_coords']['X3']
wx4 = json_data['world_coords']['X4']
wy1 = json_data['world_coords']['Y1']
wy2 = json_data['world_coords']['Y2']
wy3 = json_data['world_coords']['Y3']
wy4 = json_data['world_coords']['Y4']
world_pts.append((654,510))
world_pts.append((654,296))
world_pts.append((697,296))
world_pts.append((697,510))


# draw world pts
world_view = cv2.circle(world_img, world_pts[1], 4, (0, 0, 255), 2)
world_view = cv2.circle(world_img, world_pts[2], 4, (0, 255, 255), 2)
world_view = cv2.circle(world_img, world_pts[3], 4, (0, 255, 0), 2)
world_view = cv2.circle(world_img, world_pts[4], 4, (255, 0, 0), 2)


# save images & resize
for i in src_list:
    index = src_list.index(i)
    img = cv2.imread(src_path + i)
    src_raw.append(img)
    img_rsz = cv2.resize(img, (0, 0), fx=resize_rate, fy=resize_rate, interpolation=cv2.INTER_AREA)
    cv2.putText(img_rsz, str(index), (10,25), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255) )

    # draw pts
    img_pts = cv2.circle(img_rsz, (int(src_points[index][1][0] * resize_rate), int(src_points[index][1][1])), 2, (0, 0, 255), -1)
    img_pts = cv2.circle(img_rsz, (int(src_points[index][2][0] * resize_rate), int(src_points[index][2][1])), 2, (0, 255, 255), -1)
    img_pts = cv2.circle(img_rsz, (int(src_points[index][3][0] * resize_rate), int(src_points[index][3][1])), 2, (0, 255, 0), -1)
    img_pts = cv2.circle(img_rsz, (int(src_points[index][4][0] * resize_rate), int(src_points[index][4][1])), 2, (255, 0, 0), -1)
    src.append(img_pts)


# tile to multiview
def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


single_view = [[0 for j in range(multiview_row)] for i in range(multiview_row)]
def makeMultiView():
    cnt_view = 0
    for r in range(multiview_row):
        for c in range(multiview_row):
            if cnt_view < src_num:
                single_view[r][c] = src[cnt_view]
            else:   # black screen
                single_view[r][c] = np.zeros((src[0].shape[0], src[0].shape[1], 3), dtype=src[0].dtype)
            cnt_view += 1

makeMultiView()
multi_view = concat_tile(single_view)


# show windows
key_command = " Single Main View " \
              ": Press 'cam number' + 'Enter'. "
cv2.putText(multi_view, key_command, (100,multi_view.shape[0]-200), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0))
cv2.imshow("multi view", multi_view)
cv2.imshow("world view", world_view)
#cv2.waitKey(0)

single_index = 0
main_single_view = src_raw[single_index]
cv2.putText(main_single_view, str(single_index), (30, 80), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 255))
cv2.imshow("main view", main_single_view)


cur_point = Point()
warp_point = Point()
cntClick = 0

# mouse event
def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cur_point.x = x
        cur_point.y = y
        drawWarpPoint(cur_point)
        #setWorldPts(cur_point, cntClick)

cv2.namedWindow("main view")
cv2.setMouseCallback("main view", onMouse)
cv2.setMouseCallback("world view", onMouse)
cv2.setMouseCallback("multi view", onMouse)



def setWorldPts(pts, cnt):
    if cnt>3:
        cnt = 0
    world_pts[cnt] = pts
    color = (255,255,255)
    if cnt ==0:
        color = (0,0,255)
    elif cnt ==1:
        color = (0,255,255)
    elif cnt ==2:
        color = (0,255,0)
    elif cnt ==3:
        color = (255,0,0)

    cv2.circle(world_view, (pts.x, pts.y), 2 ,color, 1)
    cv2.imshow("world view", world_view)

    cnt += 1


color_set = (200,200,100)
def drawWarpPoint(cur_point):
    color = np.random.randint(155, 255, size=(3,))
    color = (int(color[0]), int(color[1]), int(color[2]))
    warp_p = calcPointHomo(h_sw[single_index], cur_point)
    cv2.circle(main_single_view, (cur_point.x, cur_point.y), 5, color_set, 2)
    cv2.circle(world_view, warp_p, 3, color_set, 2)
    # cv2.circle(world_view, warp_p, 3, tuple(color), 2)
    # cv2.circle(world_view, warp_p, 10, (90, 155, 100), 3)
    # cv2.circle(main_single_view, (x, y), 10, (0, 255, 200), 3)
    cv2.imshow("main view", main_single_view)
    cv2.imshow("world view", world_view)

    # to multiview
    wp = Point()
    wp.x = warp_p[0]
    wp.y = warp_p[1]
    warp_p_multi = []
    for i in range(src_num):
        warp_single_p = calcPointHomo(h_ws[i], wp)
        #print("warp_single_p ", i, "   ", warp_single_p)
        warp_p_multi.append(warp_single_p)
        cv2.circle(src[i], (int(warp_single_p[0] * resize_rate),(int(warp_single_p[1]* resize_rate))), 2, color_set, 3)
    makeMultiView()
    multi_view = concat_tile(single_view)
    cv2.imshow("multi view", multi_view)


# calc warp point
dst_pts = []
world_points = list()
for i in range(1,5):
    p = Point()
    p.x = world_pts[i][0]
    p.y = world_pts[i][1]
    world_points.append(p)
dst_pts = np.array([[e.x,e.y] for e in world_points])
# print("dst pts: ", dst_pts)


def findHomo(cam_idx):
    points = list()
    for i in range(1,5):
        p = Point()
        p.x = src_points[cam_idx][i][0]
        p.y = src_points[cam_idx][i][1]
        points.append(p)
    src_pts = np.array([[e.x,e.y] for e in points])
    # print(src_pts)
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    # print("homo matrix: ", H)
    return H

def findHomo_wtos(cam_idx):
    points = list()
    for i in range(1,5):
        p = Point()
        p.x = src_points[cam_idx][i][0]
        p.y = src_points[cam_idx][i][1]
        points.append(p)
    src_pts = np.array([[e.x,e.y] for e in points])
    # print(src_pts)
    H, status = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)
    # print("homo matrix: ", H)
    return H

def calcPointHomo(H, pts):
    x = (H[0][0] * pts.x + H[0][1] * pts.y + H[0][2]) / (H[2][0]*pts.x + H[2][1]*pts.y + 1)
    y = (H[1][0] * pts.x + H[1][1] * pts.y + H[1][2]) / (H[2][0]*pts.x + H[2][1]*pts.y + 1)
    return (int(x),int(y))

h_sw = [[0 for j in range(9)] for i in range(src_num)]
for i in range(src_num):
    h_sw[i] = findHomo(i)

h_ws = [[0 for j in range(9)] for i in range(src_num)]
for i in range(src_num):
    h_ws[i] = findHomo_wtos(i)


# 3d world view
cv2.namedWindow("3d world view")
c_h = 4
c_d = 3
world_l = 1200

cv2.createTrackbar("dx", "3d world view", 0, 400, lambda x:x)
cv2.createTrackbar("dy", "3d world view",0 , 700,lambda x:x)

cv2.setTrackbarPos("dx", "3d world view", 150)
cv2.setTrackbarPos("dy", "3d world view", 300)

world_rsz = cv2.resize(world_view,(1600,1600))
world_cut = world_rsz[400:world_l, 0:1600].copy()
#after cutting
wp1 = [0,0]
wp2 = [world_cut.shape[1], wp1[1]]
wp3 = [wp1[0], world_cut.shape[0]]
wp4 = [wp2[0], wp3[1]]

def view3Dworld():

    dx = cv2.getTrackbarPos("dx", "3d world view")
    dy = cv2.getTrackbarPos("dy", "3d world view")

    p1 = np.float32([wp1, wp2, wp3, wp4])  # plane
    p2 = np.float32([[wp1[0]+dx, wp1[1]+dy],
                       [wp2[0]-dx, wp2[1]+dy],
                       wp3,
                       wp4])  # perspective
    mtrx = cv2.getPerspectiveTransform(p1, p2)
    return mtrx, dx, dy


# key event
while True:
    m, _x, _y = view3Dworld()
    world_rsz = cv2.resize(world_view, (1600, 1600))
    world_cut = world_rsz[400:world_l, 0:1600].copy()
    world_3d = cv2.warpPerspective(world_cut, m, (1600,800))
    world_3d_cut = world_3d[_y:wp3[1],0:1600].copy()
    cv2.imshow("3d world view", world_3d_cut)


    key = cv2.waitKey(0)
    if key &0xFF == 27:
        break


    if key == ord('a'):
        if single_index<1:
            single_index = src_num-1
        else:
            single_index -= 1
    elif key == ord('d'):
        if single_index>=src_num-1:
            single_index = 0
        else:
            single_index += 1

    if key == 49:
        color_set = (125,20,255)
    elif key == 50:
        color_set = (0,70,255)
    elif key == 51:
        color_set = (0, 250, 122)
    elif key == 52:
        color_set = (200, 0, 190)
    main_single_view = src_raw[single_index]
    cv2.putText(main_single_view, str(single_index), (30, 80), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 255))
    cv2.imshow("main view", main_single_view)

    if key == ord('s'):
        cv2.imwrite("world_result.png", world_view)



cv2.destroyAllWindows()
