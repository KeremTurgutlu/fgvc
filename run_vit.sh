python train_plant_vit.py --size 384  --patch_size 16 --patch_stride 16 --bs 16 --lr 1e-4 --epochs 24 --checkpoint False --checkpoint_nchunks 0 
python train_plant_vit.py --size 384  --patch_size 32 --patch_stride 16 --bs 16 --lr 1e-4 --epochs 24 --checkpoint False --checkpoint_nchunks 0 
python train_plant_vit.py --size 384  --patch_size 32 --patch_stride 32 --bs 32 --lr 1e-4 --epochs 24 --checkpoint False --checkpoint_nchunks 0 
python train_plant_vit.py --size 384  --patch_size 64 --patch_stride 32 --bs 32 --lr 1e-4 --epochs 24 --checkpoint False --checkpoint_nchunks 0 
python train_plant_vit.py --size 384  --patch_size 64 --patch_stride 64 --bs 32 --lr 1e-4 --epochs 24 --checkpoint False --checkpoint_nchunks 0 
