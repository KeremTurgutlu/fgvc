from fastai.vision.all import *
from torch.utils.checkpoint import checkpoint
from self_supervised.layers import *

from .custom_vit import *
from .attention import *
from .object_crops import *
from .part_crops import *

class FullImageEncoder(Module):
    "Encoder which takes whole image input then outputs attention weights + layer features"
    def __init__(self, pretrained_vit_encoder, nblocks=12, checkpoint_nchunks=2, return_attn_wgts=True):
                
        # initialize params with warm up model
        self.patch_embed = pretrained_vit_encoder.patch_embed
        self.cls_token = pretrained_vit_encoder.cls_token
        self.pos_embed = pretrained_vit_encoder.pos_embed
        self.pos_drop = pretrained_vit_encoder.pos_drop
        
        # until layer n-1, can be changed (memory trade-off)
        self.blocks = pretrained_vit_encoder.blocks[:nblocks]        
        
        # not needed now
#         self.norm = pretrained_vit_encoder.norm
        
        # gradient checkpointing
        self.checkpoint_nchunks = checkpoint_nchunks
        
        self.return_attn_wgts = return_attn_wgts
         
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # collect attn_wgts from all layers
        attn_wgts = []
        if self.return_attn_wgts:
            for i,blk in enumerate(self.blocks):
                if i<self.checkpoint_nchunks: x,attn_wgt = checkpoint(blk, x)
                else:                         x,attn_wgt = blk(x)
                attn_wgts.append(attn_wgt)
            return x,attn_wgts
        
        else:
            for i,blk in enumerate(self.blocks):
                if i<self.checkpoint_nchunks: x,_ = checkpoint(blk, x)
                else:                         x,_ = blk(x)
            return x
        
    def forward(self, x):
        return self.forward_features(x)

    

class MultiCropViT(Module):
    "Multi Scale Multi Crop ViT Model"
    def __init__(self, 
                 encoder, 
                 input_res = 384,
                 high_res = 448,
                 min_obj_area=64*64,
                 crop_sz = 128,
                 p_attn_erasing = 0.5, 
                 attn_erasing_thresh = 0.7,
                 encoder_nblocks=12,
                 checkpoint_nchunks=12):
        
        self.input_res = input_res
        self.high_res = high_res
        self.min_obj_area = min_obj_area
        self.crop_sz = crop_sz
        self.p_attn_erasing = p_attn_erasing
        self.attn_erasing_thresh = attn_erasing_thresh
        
        self.image_encoder = FullImageEncoder(encoder, nblocks=encoder_nblocks, checkpoint_nchunks=checkpoint_nchunks)
        self.norm = partial(nn.LayerNorm, eps=1e-6)(768)
        self.classifier = create_cls_module(768*4, 120, lin_ftrs=[768], use_bn=False, first_bn=False, ps=0.)
    
        
    def forward(self, xb_high_res):

        
        # get full image attention weigths / feature
        xb_input_res = F.interpolate(xb_high_res, size=(self.input_res,self.input_res))
        _, attn_wgts = self.image_encoder(xb_input_res)
        
        # get attention maps
        attention_maps = generate_batch_attention_maps(attn_wgts, None, mode=None).detach()
        
        # resize attention maps to high res
        attention_maps_high_res = F.interpolate(attention_maps[None,...],
                                                   mode='bilinear',
                                                   align_corners=True,
                                                   size=(self.high_res,self.high_res))[0]
        # resize attention maps to high res input
        attention_maps_input_res = F.interpolate(attention_maps[None,...],
                                           mode='bilinear',
                                           align_corners=True,
                                           size=(self.input_res,self.input_res))[0]
        

        # randomly apply attention erasing to full images and replace full image features
        attn_erasing_mask = (attention_maps_input_res>self.attn_erasing_thresh).unsqueeze(1)
        ps = torch.zeros(attn_erasing_mask.size(0)).float().bernoulli(self.p_attn_erasing).to(attn_erasing_mask.device)
        rand_attn_erasing_mask = 1-attn_erasing_mask*ps[...,None,None,None]
        xb_input_res_erased = rand_attn_erasing_mask*xb_input_res
        x_full, _ = self.image_encoder(xb_input_res_erased)

        

        # get object bboxes
        batch_object_bboxes = [generate_attention_coordinates(attn_map, 
                                                                num_bboxes=1,
                                                                thresh_method='mean',
                                                                min_area=self.min_obj_area,
                                                                max_area=self.high_res*self.high_res)
                                 for attn_map in attention_maps_high_res.cpu().numpy()]
        batch_object_bboxes = np.vstack(batch_object_bboxes)

        # crop objects from high res image and attention maps
        xb_input_res_objects = []
        attention_maps_input_res_objects = []

        for i, obj_bbox in enumerate(batch_object_bboxes):
            minr, minc, maxr, maxc = obj_bbox
            xb_input_res_objects             += [F.interpolate(xb_high_res[i][:,minr:maxr,minc:maxc][None,...].float(), 
                                                   size=(self.input_res,self.input_res))[0]]
            attention_maps_input_res_objects += [F.interpolate(attention_maps_high_res[i][minr:maxr,minc:maxc][None,None,...].float(), 
                                                   size=(self.input_res,self.input_res))[0][0]]

        xb_input_res_objects = torch.stack(xb_input_res_objects)
        attention_maps_input_res_objects = torch.stack(attention_maps_input_res_objects)


        # randomly apply attention erasing to objects and compute object features
        attn_erasing_mask = (attention_maps_input_res_objects>self.attn_erasing_thresh).unsqueeze(1)
        ps = torch.zeros(attn_erasing_mask.size(0)).float().bernoulli(self.p_attn_erasing).to(attn_erasing_mask.device)
        rand_attn_erasing_mask = 1-attn_erasing_mask*ps[...,None,None,None]
        xb_input_res_objects_erased = rand_attn_erasing_mask*xb_input_res_objects
        x_object, _ = self.image_encoder(xb_input_res_objects_erased)
        
        
    
        
        # get 2 crop bboxes per object
        downsampled_attention_maps_objects = F.interpolate(attention_maps_input_res_objects[None,], size=(self.input_res//3,self.input_res//3))[0]
        batch_crop_bboxes = generate_batch_crops(downsampled_attention_maps_objects.cpu(),
                                                 source_sz=self.input_res//3, 
                                                 targ_sz=self.input_res, 
                                                 targ_bbox_sz=self.crop_sz,
                                                 num_bboxes=2,
                                                 nms_thresh=0.1)

        # crop parts
        xb_input_res_crops1,xb_input_res_crops2 = [],[]
        for i, crop_bboxes in enumerate(batch_crop_bboxes):
            minr, minc, maxr, maxc = crop_bboxes[0]
            xb_input_res_crops1 += [F.interpolate(xb_input_res_objects[i][:,minr:maxr,minc:maxc][None,...].float(),
                                                  size=(self.input_res,self.input_res))[0]]
            minr, minc, maxr, maxc = crop_bboxes[1]
            xb_input_res_crops2 += [F.interpolate(xb_input_res_objects[i][:,minr:maxr,minc:maxc][None,...].float(),
                                                  size=(self.input_res,self.input_res))[0]]
        xb_input_res_crops1 = torch.stack(xb_input_res_crops1)
        xb_input_res_crops2 = torch.stack(xb_input_res_crops2)
        
        # extract crop features
        x_crops1, _ = self.image_encoder(xb_input_res_crops1)
        x_crops2, _ = self.image_encoder(xb_input_res_crops2)
        
        
#         # save for visualization
#         i = np.random.choice(range(len(xb_high_res)))
#         self.random_images = to_detach([xb_high_res[i],
#                                         xb_input_res_objects[i].float(), 
#                                         xb_input_res_crops1[i].float(), 
#                                         xb_input_res_crops2[i].float()])

        
        # concat and classify
        x = torch.cat([self.norm(x_full)[:,0],
                       self.norm(x_object)[:,0],
                       self.norm(x_crops1)[:,0],
                       self.norm(x_crops2)[:,0]], dim=-1)
        out_concat = self.classifier(x)
        return out_concat