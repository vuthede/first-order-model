import cv2
import matplotlib.pyplot as plt
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '' # cpu
import numpy as np
import time
import sys


sys.path.insert(0, "../../PRNet")
from api import PRN
prn = PRN(is_dlib = True, prefix="../../PRNet")
from utils.render import render_texture
from utils.rotate_vertices import frontalize
from utils.estimate_pose import estimate_pose



sys.path.insert(0, "../../face-parsing.PyTorch")
from model import BiSeNet
sys.path.insert(0, "../../generative_inpainting")
import torch
import os.path as osp
from PIL import Image
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import dlib
sys.path.insert(0, "../../deepfeatinterp")
import alignface


sys.path.insert(0, "../../Face-Super-Resolution")
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
from models.SRGAN_model import SRGANModel




#################################### Face Parsing##############################################
class FaceParsing():
    def __init__(self, checkpoint):
        n_classes = 19
        self.net = BiSeNet(n_classes=n_classes)
        # net.cuda()
        self.net.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
        self.net.eval()
        self.to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __inference(self, image):
        with torch.no_grad():
            cv2.imwrite("tmp.png", image)
            img = Image.open("tmp.png")
            image = img.resize((512, 512), Image.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            # img = img.cuda()
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            return parsing
    
    def get_face_mask_512(self, image):
        assert image.shape[:2]==(512,512), f'Shape of input image should be 512x512'
        parsing_anno = self.__inference(image)
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        num_of_class = np.max(vis_parsing_anno)


        facemask = np.zeros(image.shape)
        for pi in [1,2,3,4,5,6,7,8,9,10,11,12,13,17]:
            index = np.where(vis_parsing_anno == pi)
            facemask[index[0], index[1], :] = [255,255,255]
        return facemask

    def get_face_mask_512_including_neck(self, image):
        assert image.shape[:2]==(512,512), f'Shape of input image should be 512x512'
        parsing_anno = self.__inference(image)
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        num_of_class = np.max(vis_parsing_anno)


        facemask = np.zeros(image.shape)
        for pi in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,17]:
            index = np.where(vis_parsing_anno == pi)
            facemask[index[0], index[1], :] = [255,255,255]
        return facemask

    def get_face_mask(self, image, bbox, apply_dilation=True):
        mask = np.zeros(image.shape)
        x1, y1, x2, y2 = map(int,bbox)
        crop = image[y1:y2, x1:x2]
        crop = cv2.resize(crop, (512, 512))
        facemask_crop = self.get_face_mask_512(crop)
        facemask_crop = cv2.resize(facemask_crop, (x2-x1, y2-y1))
        mask[y1:y2, x1:x2] = facemask_crop

        if apply_dilation:
            kernel = np.ones((7,7),np.uint8)
            mask = cv2.dilate(mask,kernel,iterations = 5)

        return mask
    
    def get_face_mask_including_neck(self, image, bbox, apply_dilation=True):
        mask = np.zeros(image.shape)
        x1, y1, x2, y2 = map(int,bbox)
        crop = image[y1:y2, x1:x2]
        crop = cv2.resize(crop, (512, 512))
        facemask_crop = self.get_face_mask_512_including_neck(crop)
        facemask_crop = cv2.resize(facemask_crop, (x2-x1, y2-y1))
        mask[y1:y2, x1:x2] = facemask_crop

        if apply_dilation:
            kernel = np.ones((7,7),np.uint8)
            mask = cv2.dilate(mask,kernel,iterations = 5)

        return mask

    def get_part_mask_by_id_512(self, image, id):
        assert image.shape[:2]==(512,512), f'Shape of input image should be 512x512'
        parsing_anno = self.__inference(image)
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        num_of_class = np.max(vis_parsing_anno)


        facemask = np.zeros(image.shape)
        index = np.where(vis_parsing_anno == id)
        facemask[index[0], index[1], :] = [255,255,255]
        return facemask

    def get_part_mask_by_id(self, image, bbox, id, apply_dilation=True):
        mask = np.zeros(image.shape)
        x1, y1, x2, y2 = map(int,bbox)
        crop = image[y1:y2, x1:x2]
        crop = cv2.resize(crop, (512, 512))
        facemask_crop = self.get_part_mask_by_id_512(crop, id)
        facemask_crop = cv2.resize(facemask_crop, (x2-x1, y2-y1))
        mask[y1:y2, x1:x2] = facemask_crop

        if apply_dilation:
            kernel = np.ones((7,7),np.uint8)
            mask = cv2.dilate(mask,kernel,iterations = 5)
        
        return mask
    
    def paint_part_again(self, image, inpainted_image, bbox, part_id):
        part_mask = self.get_part_mask_by_id(image, bbox, part_id, apply_dilation=False)
        assert image.shape == inpainted_image.shape
        cv2.imshow("Neck mask", cv2.resize(part_mask, None, fx=0.5, fy=0.5))
        cv2.imshow("inpainted_image before", cv2.resize(inpainted_image, None, fx=0.5, fy=0.5))

        inpainted_image[np.where(part_mask==255)] = image[np.where(part_mask==255)]
        cv2.imshow("inpainted_image after", cv2.resize(inpainted_image, None, fx=0.5, fy=0.5))



        return inpainted_image
# #################################### Face Parsing##############################################


# ####################################  Generative Inpaiinting #################################
import cv2
import tensorflow as tf
import neuralgym as ng
from inpaint_model import InpaintCAModel


class GenerativeInpainting():
   
    def __init__(self, checkpoint_dir, config):
        self.model = InpaintCAModel()
        self.checkpoint_dir = checkpoint_dir
        self.FLAGS = ng.Config(config)

    def inpaint(self, image, mask):
        assert image.shape == mask.shape

        h, w, _ = image.shape
        grid = 8
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)


        sess_config = tf.ConfigProto()
        # sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            input_image = tf.constant(input_image, dtype=tf.float32)
            output = self.model.build_server_graph(self.FLAGS, input_image, reuse=tf.AUTO_REUSE)
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)
            # load pretrained model
            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            for var in vars_list:
                vname = var.name
                from_name = vname
                try:
                    var_value = tf.contrib.framework.load_variable(self.checkpoint_dir, from_name)
                    assign_ops.append(tf.assign(var, var_value))
                except:
                    print("There is something wrong.")

            sess.run(assign_ops)
            print('Model loaded.')
            result = sess.run(output)
            return result[0]



# ####################################  Generative Inpaiinting #################################


def get_crop_bbox(img, x1,y1,x2,y2):
    return img[y1:y2, x1:x2]

model_path = "/home/vuthede/Downloads/shape_predictor_81_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)
def get_landmark(im1):
    det0 = detector(im1, 0)[0]
    shape = predictor(im1, det0)
    landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
    landmarks = np.array(landmarks)
    return landmarks


def align2D(img_in, img_ref):
    """
    Find transformation from ref-->in image
    and apply it to ref image 
    81 landmarks
    """

    lm1 = get_landmark(img_ref)
    lm2 = get_landmark(img_in)

    M,loss=alignface.fit_face_landmarks(lm1[:,::-1],lm2[:,::-1], landmarks=list(range(81)), scale_landmarks=[0,16],location_landmark=30,image_dims=img_in.shape[:2])

    warp_image = alignface.warp_to_template(img_ref,M,border_value=(0.5,0.5,0.5),image_dims=img_in.shape)


    r = cv2.boundingRect(np.float32(lm2))    
    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))

    return warp_image, center



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
    # output = cv2.seamlessClone((new_image*255).astype(np.uint8),(img_in).astype(np.uint8), (facemask*255).astype(np.uint8), center, cv2.NORMAL_CLONE)

    return new_image, facemask, center


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
    # parser.add_argument('--pretrain_model_G', type=str, default='/home/vuthede/10000_G_downsample.pth')
    # parser.add_argument('--pretrain_model_G', type=str, default='/home/vuthede/10000_G_datarefined.pth')
    # parser.add_argument('--pretrain_model_G', type=str, default='/home/vuthede/90000_G_higher_pixel_weight.pth')



    parser.add_argument('--pretrain_model_D', type=str, default=None)

    args = parser.parse_args()

    return args


_transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                      std=[0.5, 0.5, 0.5])])


class FaceSuperRes():
    def __init__(self, args):
        self.sr_model =  SRGANModel(args, is_train=False)
        self.sr_model.load()
    
    def do_super_res_128(self, img_128):
        assert img_128.shape[:2] == (128,128), f'Input image shape should be (128,128). Got {img_128.shape[:2]}'
        img_128 = cv2.cvtColor(img_128, cv2.COLOR_BGR2RGB)
        input_img = torch.unsqueeze(_transform(Image.fromarray(img_128)), 0)
        self.sr_model.var_L = input_img.to(self.sr_model.device)
        self.sr_model.test()
        output_img = self.sr_model.fake_H.squeeze(0).cpu().numpy()
        output_img = np.clip((np.transpose(output_img, (1, 2, 0)) / 2.0 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

        return output_img





# Deeo models: Face Parsing, inpainter, super res
sr_model = FaceSuperRes(args=get_FaceSR_opt())
faceParsing = FaceParsing(checkpoint="../../face-parsing.PyTorch/res/cp/79999_iter.pth")
inpainter = GenerativeInpainting(checkpoint_dir="../../generative_inpainting/model_logs/release_places2_256_deepfill_v2",
                            config="../../generative_inpainting/inpaint.yml")


def inpaint_face(img, bbox):
    # Get face mask
    x1,y1,x2,y2 = map(int, bbox)
    mask = faceParsing.get_face_mask(img, bbox=(x1,y1,x2,y2))
    
    h,w,_ = img.shape
    # Inpaint face area
    # Resize a bit to inpaint, to reduce time
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    mask = cv2.resize(mask, None, fx=0.5, fy=0.5)
    result = inpainter.inpaint(img, mask)
    img = cv2.resize(img, None, fx=2, fy=2)
    mask = cv2.resize(mask, None, fx=2, fy=2)

    result = cv2.resize(result, (img.shape[1], img.shape[0]))

    return result[:,:,::-1]

def inpaint_face_fancy(img, bbox):
    # Get face mask
    x1,y1,x2,y2 = map(int, bbox)
    mask = faceParsing.get_face_mask(img, bbox=(x1,y1,x2,y2), apply_dilation=False)
    
    h,w,_ = img.shape
    # Inpaint face area
    # Resize a bit to inpaint, to reduce time
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    mask = cv2.resize(mask, None, fx=0.5, fy=0.5)
    result = inpainter.inpaint(img, mask)
    img = cv2.resize(img, None, fx=2, fy=2)
    mask = cv2.resize(mask, None, fx=2, fy=2)

    result = cv2.resize(result, (img.shape[1], img.shape[0]))

    return result[:,:,::-1]

def inpaint_face_including_neck(img, bbox):
        # Get face mask
    x1,y1,x2,y2 = map(int, bbox)
    mask = faceParsing.get_face_mask_including_neck(img, bbox=(x1,y1,x2,y2))
    h,w,_ = img.shape
    
    # Inpaint face area
    # Resize a bit to inpaint, to reduce time
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    mask = cv2.resize(mask, None, fx=0.5, fy=0.5)
    result = inpainter.inpaint(img, mask)
    img = cv2.resize(img, (w, h))
    mask = cv2.resize(mask, (w, h))

    result = cv2.resize(result, (w, h))
    result = result[:,:,::-1]   
    # result = faceParsing.paint_part_again(img, result, bbox, 14)

    return result

def pipeline(img_hd, img_syn, bbox):
    # Inpaint face in Hd image
    img_hd_face_inpainted = inpaint_face(img_hd, bbox)
    x1,y1,x2,y2 = map(int, bbox)

    # Align 2D synthesis face to cropped HD face
    crophd = get_crop_bbox(img_hd, x1, y1, x2, y2)
    crophd = cv2.resize(crophd, (512,512))
    img_syn = cv2.resize(img_syn, (512,512))

    print(img_syn.shape)
    # warp2d, center = align2D(crophd, img_syn)
    warp2d = img_syn.copy()

    # cv2.imshow("CropHd", crophd)
    # cv2.imshow("warp2d", warp2d)

    warp2d_128 = cv2.resize(warp2d, (128,128))
    warp2d = sr_model.do_super_res_128(warp2d_128)

    # cv2.imshow("warp2d super res", warp2d)


    warp2d_facemask = faceParsing.get_face_mask_512(warp2d).astype(np.uint8)
    
    crophd_inpainted = get_crop_bbox(img_hd_face_inpainted, x1, y1, x2, y2)
    # cv2.imshow("Cropinpaint", crophd_inpainted)

    crophd_inpainted = cv2.resize(crophd_inpainted, (512,512))
    # center = (center[0], center[1]-30)

    nose_in_hd = get_landmark(crophd)[30] # Nose
    center = tuple(nose_in_hd)

    kernel = np.ones((7,7),np.uint8)
    warp2d_facemask = cv2.dilate(warp2d_facemask,kernel,iterations = 3)

    final_image = cv2.seamlessClone(warp2d, crophd_inpainted, warp2d_facemask, center, cv2.NORMAL_CLONE)
    # final_image = crophd_inpainted.copy()
    # final_image[np.where(warp2d_facemask==255)] = warp2d[np.where(warp2d_facemask==255)]
    final_image = final_image.astype(np.uint8)
    final_image = cv2.resize(final_image, (x2-x1, y2-y1))

    # cv2.imshow("final_image", final_image)

    img_hd_face_inpainted[y1:y2, x1:x2] = final_image

    return img_hd_face_inpainted

def pipelinefixcenter(img_hd, img_syn, bbox, center=None):
    # Inpaint face in Hd image
    img_hd_face_inpainted = inpaint_face(img_hd, bbox)
    x1,y1,x2,y2 = map(int, bbox)

    # Align 2D synthesis face to cropped HD face
    crophd = get_crop_bbox(img_hd, x1, y1, x2, y2)
    crophd = cv2.resize(crophd, (512,512))
    img_syn = cv2.resize(img_syn, (512,512))

    warp2d = img_syn.copy()

    # cv2.imshow("CropHd", crophd)
    # cv2.imshow("warp2d", warp2d)

    warp2d_128 = cv2.resize(warp2d, (128,128))
    warp2d = sr_model.do_super_res_128(warp2d_128)

    # cv2.imshow("warp2d super res", warp2d)


    warp2d_facemask = faceParsing.get_face_mask_512(warp2d).astype(np.uint8)
    
    crophd_inpainted = get_crop_bbox(img_hd_face_inpainted, x1, y1, x2, y2)
    # cv2.imshow("Cropinpaint", crophd_inpainted)

    crophd_inpainted = cv2.resize(crophd_inpainted, (512,512))
    # center = (center[0], center[1]-30)

    if center is None:
        nose_in_hd = get_landmark(crophd)[30]
        center = tuple(nose_in_hd)

    kernel = np.ones((7,7),np.uint8)
    warp2d_facemask = cv2.dilate(warp2d_facemask,kernel,iterations = 3)

    final_image = cv2.seamlessClone(warp2d, crophd_inpainted, warp2d_facemask, center, cv2.NORMAL_CLONE)
    # final_image = crophd_inpainted.copy()
    # final_image[np.where(warp2d_facemask==255)] = warp2d[np.where(warp2d_facemask==255)]
    final_image = final_image.astype(np.uint8)
    final_image = cv2.resize(final_image, (x2-x1, y2-y1))

    # cv2.imshow("final_image", final_image)

    img_hd_face_inpainted[y1:y2, x1:x2] = final_image

    return img_hd_face_inpainted, center

def pipelinefixnosestable(img_hd, img_syn, bbox, nosehd=None):
    """
    Assuming the person on synthesis video dont move the head around
    So that we can apply translation to sync between the nose of person in synthesis video and that of person
    in HD video 
    """
    # Inpaint face in Hd image
    img_hd_face_inpainted = inpaint_face(img_hd, bbox)
    # img_hd_face_inpainted = inpaint_face_including_neck(img_hd, bbox)
    x1,y1,x2,y2 = map(int, bbox)

    # Align 2D synthesis face to cropped HD face
    crophd = get_crop_bbox(img_hd, x1, y1, x2, y2)
    crophd = cv2.resize(crophd, (512,512))
    img_syn = cv2.resize(img_syn, (512,512))

    warp2d = img_syn.copy()

    # cv2.imshow("CropHd", crophd)
    # cv2.imshow("warp2d", warp2d)

    warp2d_128 = cv2.resize(warp2d, (128,128))
    warp2d = sr_model.do_super_res_128(warp2d_128)

    # cv2.imshow("warp2d super res", warp2d)


    warp2d_facemask = faceParsing.get_face_mask_512(warp2d).astype(np.uint8)
    
    crophd_inpainted = get_crop_bbox(img_hd_face_inpainted, x1, y1, x2, y2)
    # cv2.imshow("Cropinpaint", crophd_inpainted)

    crophd_inpainted = cv2.resize(crophd_inpainted, (512,512))
    # center = (center[0], center[1]-30)

    kernel = np.ones((7,11),np.uint8)
    warp2d_facemask = cv2.dilate(warp2d_facemask,kernel,iterations = 3)


    if nosehd is None: # Get the nose landmark in the first frame and then use it later
        nosehd = get_landmark(crophd)[30] + np.array([0,20])
        # nosehd = get_landmark(warp2d)[30]

    
    # Find the center so that the nose of the synthesis face will align mostly exactly the same with the HD video
    # dont move
    index = np.where(warp2d_facemask==255)[:2][::-1] # Get array([x1,y1],..[xn,yn])
    xmin, ymin = np.min(index, axis=1)
    xmax, ymax = np.max(index, axis=1)
    mid = [(xmin+xmax)//2, (ymin+ymax)//2]
    mid_crop = mid - np.array([xmin, ymin])
    nose_sythesis = get_landmark(warp2d)[30]
    nose_sythesis_crop = nose_sythesis - np.array([xmin, ymin])
    translation = mid_crop-nose_sythesis_crop 
    center = nosehd + translation
    

    final_image = cv2.seamlessClone(warp2d, crophd_inpainted, warp2d_facemask, tuple(center), cv2.NORMAL_CLONE)
    # final_image = crophd_inpainted.copy()
    # final_image[np.where(warp2d_facemask==255)] = warp2d[np.where(warp2d_facemask==255)]
    final_image = final_image.astype(np.uint8)
    final_image = cv2.resize(final_image, (x2-x1, y2-y1))

    # cv2.imshow("final_image", final_image)

    img_hd_face_inpainted[y1:y2, x1:x2] = final_image

    return img_hd_face_inpainted, nosehd

def pipeline3d(img_hd, img_syn, bbox):
    # Inpaint face in Hd image
    img_hd_face_inpainted = inpaint_face(img_hd, bbox)
    x1,y1,x2,y2 = map(int, bbox)

    # Align 2D synthesis face to cropped HD face
    crophd = get_crop_bbox(img_hd, x1, y1, x2, y2)
    crophd = cv2.resize(crophd, (256,256))

    warp3d, mask3d, center = warp_face(crophd, img_syn)

    cv2.imshow("CropHd", crophd)
    cv2.imshow("warp3d", warp3d)
    cv2.imshow("mask3d", mask3d)


    # cv2.imshow("warp2d super res", warp2d)


    # warp2d_facemask = faceParsing.get_face_mask_512(warp2d).astype(np.uint8)
    
    crophd_inpainted = get_crop_bbox(img_hd_face_inpainted, x1, y1, x2, y2)
    cv2.imshow("Cropinpaint", crophd_inpainted)
    crophd_inpainted = cv2.resize(crophd_inpainted, (512,512))
    
    final_image = crophd_inpainted.copy()
    final_image[np.where(mask3d==255)] = warp3d[np.where(mask3d==255)]
    final_image = final_image.astype(np.uint8)

    # Super res
    final_image = cv2.resize(final_image, (128,128))
    final_image = sr_model.do_super_res_128(final_image)


    final_image = cv2.resize(final_image, (x2-x1, y2-y1))

    cv2.imshow("final_image", final_image)

    img_hd_face_inpainted[y1:y2, x1:x2] = final_image

    return img_hd_face_inpainted

def pipelineneckclone(img_hd, img_syn, bbox):
    # Inpaint face in Hd image
    # img_hd_face_inpainted = inpaint_face_including_neck(img_hd, bbox)
    img_hd_face_inpainted = inpaint_face_fancy(img_hd, bbox)

    x1,y1,x2,y2 = map(int, bbox)

    # Align 2D synthesis face to cropped HD face
    crophd = get_crop_bbox(img_hd, x1, y1, x2, y2)
    crophd = cv2.resize(crophd, (512,512))
    # warp2d, center = align2D(crophd, img_syn)
    img_syn = cv2.resize(img_syn, (512,512))

    warp2d = img_syn.copy()

    # cv2.imshow("CropHd", crophd)
    # cv2.imshow("warp2d", warp2d)

    warp2d_128 = cv2.resize(warp2d, (128,128))
    warp2d = sr_model.do_super_res_128(warp2d_128)

    warp2d_facemask = faceParsing.get_face_mask_512_including_neck(warp2d).astype(np.uint8)
    
    crophd_inpainted = get_crop_bbox(img_hd_face_inpainted, x1, y1, x2, y2)
    cv2.imshow("Cropinpaint", crophd_inpainted)
    # cv2.imshow("warp2d_facemask", warp2d_facemask)


    crophd_inpainted = cv2.resize(crophd_inpainted, (512,512))
    # center = (center[0], center[1]-30)

    kernel = np.ones((7,7),np.uint8)
    warp2d_facemask = cv2.dilate(warp2d_facemask,kernel,iterations = 3)

    ray_image = cv2.cvtColor(warp2d_facemask, cv2.COLOR_BGR2GRAY)

    index = np.where(warp2d_facemask==255)[:2][::-1]
    xmin,ymin = np.min(index, axis=1)
    xmax,ymax = np.max(index, axis=1)
    center = ( (xmin+xmax)//2, (ymin+ymax)//2)
    # print(center)

    final_image = cv2.seamlessClone(warp2d, crophd_inpainted, warp2d_facemask, center, cv2.NORMAL_CLONE)
    # final_image = crophd_inpainted.copy()
    # final_image[np.where(warp2d_facemask==255)] = warp2d[np.where(warp2d_facemask==255)]
    final_image = final_image.astype(np.uint8)
    final_image = cv2.resize(final_image, (x2-x1, y2-y1))

    # cv2.imshow("final_image", final_image)

    img_hd_face_inpainted[y1:y2, x1:x2] = final_image
    # cv2.imshow("before postporcess", img_hd_face_inpainted)


    # Post processing, Get background of the old image
    # face_neck_mask =  faceParsing.get_face_mask_including_neck(img_hd_face_inpainted, bbox).astype(np.uint8)
    # background_mask = ~face_neck_mask
    # shirt_mask = cv2.imread("/home/vuthede/Desktop/mask_shirt2_fit.png")
    # shirt_mask = ~shirt_mask
    # cv2.imshow("shirt mask original", shirt_mask)

    # print("shirt mask al:", shirt_mask.shape)

    # neck_mask = faceParsing.get_part_mask_by_id(img_hd_face_inpainted, bbox, 14).astype(np.uint8)
    # neck_mask_black = ~neck_mask



    # shirt_mask = shirt_mask & neck_mask_black
    # cv2.imshow("shirt mask all", shirt_mask)


    # img_hd_tmp = img_hd.copy()
    # img_hd_tmp = cv2.resize(img_hd_tmp, None, fx=0.5, fy=0.5)
    # img_hd_tmp = cv2.resize(img_hd_tmp, (img_hd.shape[1], img_hd.shape[0]))


    # index = np.where(shirt_mask==255)[:2][::-1]
    # xmin,ymin = np.min(index, axis=1)
    # xmax,ymax = np.max(index, axis=1)
    # center = ( (xmin+xmax)//2, (ymin+ymax)//2-4)
    # print(center)

    # img_hd_face_inpainted = cv2.seamlessClone(img_hd_tmp,img_hd_face_inpainted , shirt_mask, center, cv2.NORMAL_CLONE)

   

    # print("img_hd_face_inpainted shape:", img_hd_face_inpainted.shape)
    # print("img_hd tmp shape:", img_hd_tmp.shape)
    # print("shirt mask all shape:", shirt_mask.shape)

    # img_hd_face_inpainted[np.where(shirt_mask==255)] = img_hd_tmp[np.where(shirt_mask==255)]
    # haha = cv2.resize(img_hd_face_inpainted, (512,512))
    # cv2.imshow("haha", haha)


    return img_hd_face_inpainted

def pipelineneckandface(img_hd, img_syn, bbox):
    x1,y1,x2,y2 = map(int, bbox)

    # Inpaint face and neck in Hd image 
    img_hd_face_inpainted = inpaint_face_including_neck(img_hd, bbox)
    crophd_inpainted = get_crop_bbox(img_hd_face_inpainted, x1, y1, x2, y2)
    crophd_inpainted = cv2.resize(crophd_inpainted, (512,512))

    # crop HD image and and resize them into 512x512
    crophd = get_crop_bbox(img_hd, x1, y1, x2, y2)
    crophd = cv2.resize(crophd, (512,512))
    img_syn = cv2.resize(img_syn, (512,512))
    warp2d = img_syn.copy()

    # Doing face super resolution
    warp2d_128 = cv2.resize(warp2d, (128,128))
    warp2d = sr_model.do_super_res_128(warp2d_128)


    warp2d_facemask = faceParsing.get_face_mask_512(warp2d).astype(np.uint8)
    warp2d_neckmask = faceParsing.get_part_mask_by_id_512(warp2d, 14).astype(np.uint8)

  

    # Dilation mask
    kernel = np.ones((7,7),np.uint8)
    warp2d_facemask = cv2.dilate(warp2d_facemask,kernel,iterations = 3)

    # Find the center of the mask
    index = np.where(warp2d_facemask==255)[:2][::-1]
    xmin,ymin = np.min(index, axis=1)
    xmax,ymax = np.max(index, axis=1)
    center = ( (xmin+xmax)//2, (ymin+ymax)//2)

    final_image = cv2.seamlessClone(warp2d, crophd_inpainted, warp2d_facemask, center, cv2.NORMAL_CLONE)
    final_image = final_image.astype(np.uint8)
    final_image = cv2.resize(final_image, (x2-x1, y2-y1))

    cv2.imshow("final_image", final_image)

    img_hd_face_inpainted[y1:y2, x1:x2] = final_image

    return img_hd_face_inpainted




def demo_fullhd():
    out = cv2.VideoWriter('./highresfullhd_test.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 15, (1920, 1080))
    out_images = "highresfullhd_test"
    if not os.path.isdir(out_images):
        os.makedirs(out_images)
    

    # HD video and synthesis video
    # cap = cv2.VideoCapture("../result22072020night.mp4")
    cap = cv2.VideoCapture("../result25072020mornign.mp4")
    cap_hd = cv2.VideoCapture("../obama_fullhd.mp4")

    # Warm up:
    cap_hd.set(1, 2*60*30+5)
    bbox = [685,85,1250,650]

    i = 0
    cap.set(1, 5)
    center = None
    nosehd = None
    while True:

        ret, img = cap.read()
        ret2, img_hd = cap_hd.read()

        if i %(5*30) ==0: # Loop 5s
            cap_hd.set(1, 2*60*30) # Loop fullhd

        if i%1==0:
            print("Goo.............")
            if not ret or not ret2:
                break
            
            # result = pipelineneckandface(img_hd, img, bbox)
            result = pipelineneckclone(img_hd, img, bbox)
            # result, nosehd = pipelinefixnosestable(img_hd, img, bbox, nosehd)
            # result, center = pipelinefixcenter(img_hd, img,bbox, center)
            # result = pipeline3d(img_hd, img, bbox)

            # out.write(result)

            print(result.shape)
            cv2.imwrite(f'{out_images}/{i}.png', result)

            cv2.imshow("Full Hd", result)

            k = cv2.waitKey(1)

            if k==27:
                break

        i+=1

    out.release()
    cv2.destroyAllWindows()



if __name__=="__main__":
    demo_fullhd()
   

