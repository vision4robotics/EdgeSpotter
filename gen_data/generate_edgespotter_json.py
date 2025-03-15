#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import os
import sys
import cv2
import numpy as np
from shapely.geometry import *

Basis_matrix = lambda ts: [catmull_rom_basis(t) for t in ts]
def catmull_rom_basis(t):
    t2 = t ** 2
    t3 = t ** 3
    return np.stack([
        -0.5 * t3 + t2 - 0.5 * t,  
        1.5 * t3 - 2.5 * t2 + 1.0, 
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,  
        0.5 * t3 - 0.5 * t2 
    ])

cV2 = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
root_path = 'D:\\pythonProject\\2023\\OCR_DataSet_up'
dataset = {
    'licenses': [],
    'info': {},
    'categories': [],
    'images': [],
    'annotations': []
}
with open(os.path.join(root_path, 'classes.txt')) as f:
  classes = f.read().strip().split()
for i, cls in enumerate(classes, 1):
  dataset['categories'].append({
      'id': i,
      'name': cls,
      'supercategory': 'beverage',
      'keypoints': ['mean',
                    'xmin',
                    'x2',
                    'x3',
                    'xmax',
                    'ymin',
                    'y2',
                    'y3',
                    'ymax',
                    'cross']  
  })


def get_category_id(cls):
  for category in dataset['categories']:
    if category['name'] == cls:
      return category['id']


_indexes = sorted([f.split('.')[0]
                   for f in os.listdir(os.path.join(root_path, 'IPM2025\\gt_catmull_rom'))])


indexes = [line for line in _indexes]
j = 1
for index in indexes:
  im = cv2.imread(os.path.join(root_path, 'IPM2025\\images\\') +index + '.jpg')
  height, width, _ = im.shape
  dataset['images'].append({
      'coco_url': '',
      'date_captured': '',
      'file_name': index + '.jpg',
      'flickr_url': '',
      'id': int(index.split('_')[-1]),
      'license': 0,
      'width': width,
      'height': height
  })
  anno_file = os.path.join(root_path, 'IPM2025\\gt_catmull_rom\\') + index + '.txt'
  with open(anno_file, encoding='utf-8') as f:
    lines = [line for line in f.readlines() if line.strip()]
    for i, line in enumerate(lines):
      pttt = line.strip().split('####')
      parts = pttt[0].split(',')
      ct = pttt[-1].strip()
      segs = [float(kkpart) for kkpart in parts[:16]]
      # CatmullRom sampling
      control_points = [
        [segs[0], segs[1]], [segs[2], segs[3]], [segs[4], segs[5]], [segs[6], segs[7]]
      ]
      control_points_b = [
        [segs[8], segs[9]], [segs[10], segs[11]], [segs[12], segs[13]], [segs[14], segs[15]]
      ]
      t_plot = np.linspace(0, 1, 81)
      CatmullRom_top = np.array(Basis_matrix(t_plot)).dot(control_points)

      CatmullRom_bottom = np.array(Basis_matrix(t_plot)).dot(control_points_b)

      all_points = np.concatenate((CatmullRom_top, CatmullRom_bottom), axis=0)

      xmax = np.max(all_points[:, 0])
      xmin = max(0.0, np.min(all_points[:, 0]))
      ymax = np.max(all_points[:, 1])
      ymin = max(0.0, np.min(all_points[:, 1]))
      
      width_area = max(0, xmax - xmin + 1)
      height_area = max(0, ymax - ymin + 1)
      if width_area == 0 or height_area == 0:
        continue

      max_len = 100
      recs = [len(cV2)+1 for ir in range(max_len)]
      
      ct =  str(ct)
      
      for ix, ict in enumerate(ct):        
        if ix >= max_len: continue
        if ict in cV2:
            recs[ix] = cV2.index(ict)
        else:
          recs[ix] = len(cV2)

      dataset['annotations'].append({
          'area': width_area * height_area,
          'bbox': [xmin, ymin, width_area, height_area],
          'category_id': get_category_id(cls),
          'id': j,
          'image_id': int(index.split('_')[-1]),
          'iscrowd': 0,
          'catmullrom_pts': segs,
          'rec': recs
      })
      j += 1
folder = os.path.join(root_path, 'IPM2025')
if not os.path.exists(folder):
  os.makedirs(folder)
json_name = os.path.join(root_path, 'IPM2025\\{}.json'.format('train_catmull'))
with open(json_name, 'w') as f:
  json.dump(dataset, f)
