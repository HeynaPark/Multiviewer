import cv2
import numpy as np
import os
import json
import natsort

# print(cv2.__version__)
# print(os.getcwd())

src = []
src_raw = []
#calib = None
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
world_img = cv2.imread('/home/hnpark/data/IceLinkHalf_old.png')


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
print(src_points)
world_pts.append(json_data['stadium'])
wx1 = json_data['world_coords']['X1']
wx2 = json_data['world_coords']['X2']
wx3 = json_data['world_coords']['X3']
wx4 = json_data['world_coords']['X4']
wy1 = json_data['world_coords']['Y1']
wy2 = json_data['world_coords']['Y2']
wy3 = json_data['world_coords']['Y3']
wy4 = json_data['world_coords']['Y4']
world_pts.append((int(wx1), int(wy1)))
world_pts.append((int(wx2), int(wy2)))
world_pts.append((int(wx3), int(wy3)))
world_pts.append((int(wx4), int(wy4)))


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
cnt_view = 0
for r in range(multiview_row):
    for c in range(multiview_row):
        if cnt_view < src_num:
            single_view[r][c] = src[cnt_view]
        else:   # black screen
            single_view[r][c] = np.zeros((src[0].shape[0], src[0].shape[1], 3), dtype=src[0].dtype)
        cnt_view += 1

multi_view = concat_tile(single_view)


# show windows
key_command = " Single Main View " \
              ": Press 'cam number' + 'Enter'. "
cv2.putText(multi_view, key_command, (100,multi_view.shape[0]-200), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0))
#cv2.imshow("multi view", multi_view)
cv2.imshow("world view", world_view)
#cv2.waitKey(0)

single_index = 10
main_single_view = src_raw[single_index]
cv2.putText(main_single_view, str(single_index), (30, 80), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 255))
cv2.imshow("main view", main_single_view)


cur_point = Point()
warp_point = Point()
# mouse event
def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cur_point.x = x
        cur_point.y = y
        color = np.random.randint(0, 255, size=(3, ))
        color = (int(color[0]), int(color[1]), int(color[2]))
        print(color)
        warp_p = calcPointHomo(h0, cur_point)
        print("warp point 1:" , warp_p)
        cv2.circle(world_view, warp_p, 10, tuple(color), 3)
        cv2.circle(main_single_view, (x, y), 10, color, 3)
        # cv2.circle(world_view, warp_p, 10, (90, 155, 100), 3)
        # cv2.circle(main_single_view, (x, y), 10, (0, 255, 200), 3)
        cv2.imshow("world view", world_view)
        cv2.imshow("main view", main_single_view)


cv2.namedWindow("main view")
cv2.setMouseCallback("main view", onMouse)
cv2.setMouseCallback("world view", onMouse)


# while True:
#     if cv2.waitKey(0) &0xFF == 27:
#         break


# calc warp point
dst_pts = []
world_points = list()
for i in range(1,5):
    p = Point()
    p.x = world_pts[i][0]
    p.y = world_pts[i][1]
    world_points.append(p)
dst_pts = np.array([[e.x,e.y] for e in world_points])
print("dst pts: ", dst_pts)
def findHomo(cam_idx):
    points = list()
    for i in range(1,5):
        p = Point()
        p.x = src_points[cam_idx][i][0]
        p.y = src_points[cam_idx][i][1]
        points.append(p)
    src_pts = np.array([[e.x,e.y] for e in points])
    print(src_pts)
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    print("homo matrix: ", H)
    return H
def calcPointHomo(H, pts):
    x = (H[0][0] * pts.x + H[0][1] * pts.y + H[0][2]) / (H[2][0]*pts.x + H[2][1]*pts.y + 1)
    y = (H[1][0] * pts.x + H[1][1] * pts.y + H[1][2]) / (H[2][0]*pts.x + H[2][1]*pts.y + 1)
    return (int(x),int(y))

h0 = []
h0 = findHomo(single_index)
#print(h0)

#warp_point = calcPointHomo(h0, cur_point)
#print("warp point: 2" , warp_point)



cv2.waitKey()
cv2.destroyAllWindows()


# click = False
# pos_x, pos_y = -1, -1
# def draw_rectangle(event, x, y, flags, param):
#     global pos_x, pos_y, click
#
#     if event == cv2.EVENT_LBUTTONDOWN:
#         click = True
#         pos_x , pos_y = x, y
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if click ==True:
#             cv2.rectangle(main_single_view, (pos_x, pos_y), (x,y), (255,0,0), 2)
#     elif event == cv2.EVENT_LBUTTONUP:
#         click = False
#         cv2.rectangle(main_single_view, (pos_x, pos_y), (x,y), (255,0,0), 2)
#
# cv2.namedWindow("main view")
# cv2.setMouseCallback("main view", draw_rectangle)
# cv2.imshow("main view", main_single_view)
# cv2.waitKey()


# key press event
# main_view_num = 0
#
# cam_idx = 0
# while True:
#
#     key = cv2.waitKey()
#     if key in range(48,58):
#         cam_idx = key - 48
#     elif key == 13:
#         print(cam_idx)
#     elif key == ord('q'):
#         break


# cv2.imshow("multi view", multi_view)
# #cv2.imshow("world view", world_img)
# cv2.waitKey(0)




                                        

