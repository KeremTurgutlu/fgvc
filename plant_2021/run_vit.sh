#########################
### No Slide vs Slide ###
########################

# ViT-B/16-12 and ViT-B/16-16
python train_plant_vit.py --size 384  --patch_sizes [16] --patch_strides [12] --bs 16 --lr 1e-4 --epochs 12 --checkpoint True --checkpoint_nchunks 6
python train_plant_vit.py --size 384  --patch_sizes [16] --patch_strides [16] --bs 16 --lr 1e-4 --epochs 12 --checkpoint False --checkpoint_nchunks 0 

# ViT-B/32-16 and ViT-B/32-32
python train_plant_vit.py --size 384  --patch_sizes [32] --patch_strides [16] --bs 16 --lr 1e-4 --epochs 12 --checkpoint True --checkpoint_nchunks 2
python train_plant_vit.py --size 384  --patch_sizes [32] --patch_strides [32] --bs 16 --lr 1e-4 --epochs 12 --checkpoint False --checkpoint_nchunks 0 

# ViT-B/64-32 and ViT-B/64-64
python train_plant_vit.py --size 384  --patch_sizes [64] --patch_strides [32] --bs 16 --lr 1e-4 --epochs 12 --checkpoint False --checkpoint_nchunks 0 
python train_plant_vit.py --size 384  --patch_sizes [64] --patch_strides [64] --bs 16 --lr 1e-4 --epochs 12 --checkpoint False --checkpoint_nchunks 0 

#################################
### Multi Patch              ###
################################

# ViT-B/64-64+32-32+16-16
python train_plant_vit.py --size 384  --patch_sizes [64,32,16] --patch_strides [64,32,16] --bs 16 --lr 1e-4 --epochs 12 --checkpoint True --checkpoint_nchunks 2

# ViT-B/64-64+32-32
python train_plant_vit.py --size 384  --patch_sizes [64,32] --patch_strides [64,32] --bs 16 --lr 1e-4 --epochs 12 --checkpoint True --checkpoint_nchunks 2

# ViT-B/32-32+16-16
python train_plant_vit.py --size 384  --patch_sizes [32,16] --patch_strides [32,16] --bs 16 --lr 1e-4 --epochs 12 --checkpoint True --checkpoint_nchunks 2 

#################################
### Multi Patch +  Slide     ###
################################

# ViT-B/64-32+32-16
python train_plant_vit.py --size 384  --patch_sizes [64,32] --patch_strides [32,16] --bs 16 --lr 1e-4 --epochs 12 --checkpoint True --checkpoint_nchunks 2

# ViT-B/32-32+16-16
python train_plant_vit.py --size 384  --patch_sizes [32,16] --patch_strides [16,12] --bs 16 --lr 1e-4 --epochs 12 --checkpoint True --checkpoint_nchunks 6

# ViT-B/64-64+32-32+16-16
python train_plant_vit.py --size 384  --patch_sizes [64,32,16] --patch_strides [48,24,12] --bs 16 --lr 1e-4 --epochs 12 --checkpoint True --checkpoint_nchunks 12

