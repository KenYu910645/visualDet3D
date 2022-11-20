import cv2
import pickle
import random
import torch
img = cv2.imread("/home/lab530/KenYu/kitti_mixup_1/testing/image_2/000169.png")
img = cv2.resize(img, (1280, 288))

with open("useful_mask.pkl", 'rb') as f1:
    with open("all_anchors.pkl", 'rb') as f2: # anchor 
        useful_mask = pickle.load(f1)[0]
        # print(f"useful_mask.shape = {useful_mask[0].shape}")
        all_anchors = pickle.load(f2)

        print(torch.count_nonzero(useful_mask)) # 18894

        is_useful = False
        i = 0
        idx_grid = 0
        for x1, y1, x2, y2 in all_anchors:
            if i % 32 == 0:
                color = tuple([random.randint(0, 255) for _ in range(3)])
                if is_useful:
                    # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                    is_useful = False
                idx_grid += 1
            
            # Roll a dice
            if random.random() < 0.01 and useful_mask[i]:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)


            if useful_mask[i] : is_useful = True
            # if idx_grid == 600:
            #     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
            i += 1

cv2.imwrite("tmp.png", img)