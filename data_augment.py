import cv2
import os
from tqdm import tqdm
import random

DIR = 'dataset/'
COLORS = [[200, 0, 0], [0, 200, 0], [0, 0, 200], [200, 200, 0]]
black = [0, 0, 0]
for subdir in ['train', 'val']:
    img_path = os.path.join(DIR, subdir, 'images')
    mask_path = os.path.join(DIR, subdir, 'masks')

    for img in tqdm(os.listdir(img_path)):
        image = cv2.imread(os.path.join(img_path, img))
        mask = cv2.imread(os.path.join(mask_path, os.path.splitext(img)[0] + '_mask.png'))
        (img_w, img_h) = image.shape[:-1]
        #print(image.shape[:-1])
        (obj_w, obj_h) = (random.randint(70, 100), random.randint(70, 100))
        (x, y) = (random.randint(img_w/2 - 40, img_w/2 + 40), random.randint(img_h/2 - 70, img_h/2 + 20))
        (x2, y2) = (x+obj_w, y+obj_h)
        color = random.choice(COLORS)
        cv2.rectangle(image, (x, y), (x2, y2), color, -1)
        cv2.rectangle(mask, (x, y), (x2, y2), black, -1)

        cv2.imwrite(os.path.join(img_path, img), image)
        cv2.imwrite(os.path.join(mask_path, os.path.splitext(img)[0] + '_mask.png'), mask)
