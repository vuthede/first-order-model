import cv2
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "../../PRNet")
from api import PRN
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '' # cpu
prn = PRN(is_dlib = True, prefix="../../PRNet")
from utils.render import render_texture
from utils.rotate_vertices import frontalize
from utils.estimate_pose import estimate_pose
import numpy as np
import time

def warp_face(img_in, img_ref):
    #img_in is HD image
    # img_ref is synthesis image
    # Calculate position map and get pose and rotation matrix
    pos1 = prn.process(img_in) 
    vertices1 = prn.get_vertices(pos1)
    cam_mat1, pose1, R1 = estimate_pose(vertices1)
    pos2 = prn.process(img_ref) 
    vertices2 = prn.get_vertices(pos2)
    cam_mat2, pose2, R2 = estimate_pose(vertices2)

    # Rotation 3D vertices
    warp_vertices = np.matmul(np.matmul(vertices2,R2), np.linalg.inv(R1)) 

    # Do translation
    center2_warp_pt = np.mean(warp_vertices, axis=0)
    center1_pt = np.mean(vertices1, axis=0)
    warp_vertices = warp_vertices - (center2_warp_pt - center1_pt)

    # Render new synthesis image after doing transformation
    # t1 = time.time()
    texture_ref = cv2.remap(img_ref/255.0, pos2[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    # print("Texture from pos map: ", time.time()-t1)

    [h, w, c] = img_ref.shape
    # t1 = time.time()
    
    color = prn.get_colors_from_texture(texture_ref)
    # print("Get color of texture: ", time.time()-t1)
    
    color_mask = np.ones((warp_vertices.shape[0], 1))
    t1 = time.time()
    new_image = render_texture(warp_vertices.T, color.T, prn.triangles.T, h, w, c = 3)
    print("Time render: ", time.time()-t1)
    facemask = render_texture(warp_vertices.T, color_mask.T, prn.triangles.T, h, w, c = 3)

    # Using seamlessCloning to blending images
    vis_ind = np.argwhere(facemask>0)
    vis_min = np.min(vis_ind, 0)
    vis_max = np.max(vis_ind, 0)
    center = (int((vis_min[1] + vis_max[1])/2+0.5), int((vis_min[0] + vis_max[0])/2+0.5))
    output = cv2.seamlessClone((new_image*255).astype(np.uint8),(img_in).astype(np.uint8), (facemask*255).astype(np.uint8), center, cv2.NORMAL_CLONE)

    return output

sys.path.insert(0, "../../Face-Super-Resolution")
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
from models.SRGAN_model import SRGANModel
def get_FaceSR_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_G', type=float, default=1e-4)
    parser.add_argument('--weight_decay_G', type=float, default=0)
    parser.add_argument('--beta1_G', type=float, default=0.9)
    parser.add_argument('--beta2_G', type=float, default=0.99)
    parser.add_argument('--lr_D', type=float, default=1e-4)
    parser.add_argument('--weight_decay_D', type=float, default=0)
    parser.add_argument('--beta1_D', type=float, default=0.9)
    parser.add_argument('--beta2_D', type=float, default=0.99)
    parser.add_argument('--lr_scheme', type=str, default='MultiStepLR')
    parser.add_argument('--niter', type=int, default=100000)
    parser.add_argument('--warmup_iter', type=int, default=-1)
    parser.add_argument('--lr_steps', type=list, default=[50000])
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--pixel_criterion', type=str, default='l1')
    parser.add_argument('--pixel_weight', type=float, default=1e-2)
    parser.add_argument('--feature_criterion', type=str, default='l1')
    parser.add_argument('--feature_weight', type=float, default=1)
    parser.add_argument('--gan_type', type=str, default='ragan')
    parser.add_argument('--gan_weight', type=float, default=5e-3)
    parser.add_argument('--D_update_ratio', type=int, default=1)
    parser.add_argument('--D_init_iters', type=int, default=0)

    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--val_freq', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--crop_size', type=float, default=0.85)
    parser.add_argument('--lr_size', type=int, default=128)
    parser.add_argument('--hr_size', type=int, default=512)

    # network G
    parser.add_argument('--which_model_G', type=str, default='RRDBNet')
    parser.add_argument('--G_in_nc', type=int, default=3)
    parser.add_argument('--out_nc', type=int, default=3)
    parser.add_argument('--G_nf', type=int, default=64)
    parser.add_argument('--nb', type=int, default=16)

    # network D
    parser.add_argument('--which_model_D', type=str, default='discriminator_vgg_128')
    parser.add_argument('--D_in_nc', type=int, default=3)
    parser.add_argument('--D_nf', type=int, default=64)

    # data dir
    parser.add_argument('--pretrain_model_G', type=str, default='/home/vuthede/AI/Face-Super-Resolution/checkpoints/40000_G_our.pth')
    parser.add_argument('--pretrain_model_D', type=str, default=None)

    args = parser.parse_args()

    return args


_transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                      std=[0.5, 0.5, 0.5])])


x1 = 685
y1 = 85
x2 = 1250
y2 = 650
if __name__=="__main__":
    # out = cv2.VideoWriter('./highres.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (512*3, 512))

    out = cv2.VideoWriter('./highresfullhd.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 2, (1920, 1080))

    
    sr_model = SRGANModel(get_FaceSR_opt(), is_train=False)
    sr_model.load()

    cap = cv2.VideoCapture("../resultdeenglish.mp4")
    cap_de = cv2.VideoCapture("../cropdeenglish.mp4")
    cap_hd = cv2.VideoCapture("../obama_fullhd.mp4")


    # hd_img = cv2.imread("/home/vuthede/Desktop/face2.png")
    # hd_img = cv2.resize(hd_img, (256,256))

    while True:
        ret, img = cap.read()
        ret1, img_de = cap_de.read()
        ret2, img_hd = cap_hd.read()


        # cv2.imshow("De", img_de)


        if not ret or not ret1 or not ret2:
            break
        
        img_de = cv2.resize(img_de, (512,512))

        img1 = cv2.resize(img.copy(), (512,512))
        
        # cv2.imshow("Original", img1)

        # Warp3d
        crop_hd = img_hd[y1:y2,x1:x2]
        crop_hd = cv2.resize(crop_hd, (512,512))
        # cv2.imshow("crop HD", crop_hd)

        img = warp_face(crop_hd, img1)
        # cv2.imshow("After warp", img)


        img_128 = cv2.resize(img, (128,128))
        img_128 = cv2.cvtColor(img_128, cv2.COLOR_BGR2RGB)
        input_img = torch.unsqueeze(_transform(Image.fromarray(img_128)), 0)
        sr_model.var_L = input_img.to(sr_model.device)
        sr_model.test()
        output_img = sr_model.fake_H.squeeze(0).cpu().numpy()
        output_img = np.clip((np.transpose(output_img, (1, 2, 0)) / 2.0 + 0.5) * 255.0, 0, 255).astype(np.uint8)

        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

        

        # concat = np.hstack([img_de, img1, output_img])
        # print("Concat shape:", concat.shape)
        # out.write(concat)
        # cv2.imshow("concat", concat)

        # crophd ---->full frame
        output_img = cv2.resize(output_img,  (x2-x1, y2-y1))
        img_hd[y1:y2, x1:x2] = output_img

        out.write(img_hd)

        cv2.imshow("Full Hd", img_hd)



        k = cv2.waitKey(1)

        if k==27:
            break

    out.release()
    cv2.destroyAllWindows()

