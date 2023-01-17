import cv2
import numpy as np
import os
import json
import natsort

src = []
calib = None
src_w = 1920
src_h = 1080
cam_pts = []
world_pts = []

print(cv2.__version__)
print(os.getcwd())

# src import
src_path = '/home/hnpark/data/1/'
src_list = os.listdir(src_path)
src_list = natsort.natsorted(src_list)
src_num = len(src_list)

# world img import
world_img = cv2.imread('/home/hnpark/data/IceLinkHalf_old.png')


# calc viewer row
multiview_row = 1
while 1:
    multiview_row += 1
    if multiview_row ** 2 > src_num:
        break
resize_rate = 1 / multiview_row

# json import
json_file = open('/home/hnpark/data/1.pts')
json_data = json.load(json_file)
points = [[0 for j in range(5)] for i in range(src_num)]
for i in range(src_num):
    points[i][0] = json_data['points'][i]['dsc_id']
    x1 = json_data['points'][i]['pts_3d']['X1']
    x2 = json_data['points'][i]['pts_3d']['X2']
    x3 = json_data['points'][i]['pts_3d']['X3']
    x4 = json_data['points'][i]['pts_3d']['X4']
    y1 = json_data['points'][i]['pts_3d']['Y1']
    y2 = json_data['points'][i]['pts_3d']['Y2']
    y3 = json_data['points'][i]['pts_3d']['Y3']
    y4 = json_data['points'][i]['pts_3d']['Y4']
    points[i][1] = (int(x1 * resize_rate), int(y1 * resize_rate))
    points[i][2] = (int(x2 * resize_rate), int(y2 * resize_rate))
    points[i][3] = (int(x3 * resize_rate), int(y3 * resize_rate))
    points[i][4] = (int(x4 * resize_rate), int(y4 * resize_rate))
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

# world pts
world_view = cv2.circle(world_img, world_pts[1], 4, (0, 0, 255), 2)
world_view = cv2.circle(world_img, world_pts[2], 4, (0, 255, 255), 2)
world_view = cv2.circle(world_img, world_pts[3], 4, (0, 255, 0), 2)
world_view = cv2.circle(world_img, world_pts[4], 4, (255, 0, 0), 2)

# save images & resize
for i in src_list:
    index = src_list.index(i)
    img = cv2.imread(src_path + i)
    img_rsz = cv2.resize(img, (0, 0), fx=resize_rate, fy=resize_rate, interpolation=cv2.INTER_AREA)
    #name, ext = os.path.splitext(i)
    cv2.putText(img_rsz, str(index), (10,25), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255) )
    # draw pts
    # src.append(img_rsz)
    #print(points[src_list.index(i)][1])
    img_pts = cv2.circle(img_rsz, points[index][1], 2, (0, 0, 255), -1)
    img_pts = cv2.circle(img_rsz, points[index][2], 2, (0, 255, 255), -1)
    img_pts = cv2.circle(img_rsz, points[index][3], 2, (0, 255, 0), -1)
    img_pts = cv2.circle(img_rsz, points[index][4], 2, (255, 0, 0), -1)
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
        else:
            single_view[r][c] = np.zeros((src[0].shape[0], src[0].shape[1], 3), dtype=src[0].dtype)
        cnt_view += 1

multi_view = concat_tile(single_view)

# key press event
key_command = " Single Main View " \
              ": Press 'cam number' + 'Enter'. "
cv2.putText(multi_view, key_command, (100,multi_view.shape[0]-200), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0))

cv2.imshow("multi view", multi_view)
cv2.imshow("world view", world_img)
cv2.waitKey(0)




                                        

