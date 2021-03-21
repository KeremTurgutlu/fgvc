python train_plant_baseline.py --arch tf_efficientnet_b4_ns --size 448 --lr 2e-2 --bs 32 --epochs 24
python train_plant_baseline.py --arch resnet101d            --size 448 --lr 2e-2 --bs 32 --epochs 24
python train_plant_baseline.py --arch vit_base_patch16_384  --size 384 --lr 1e-3 --bs 32 --epochs 24
python train_plant_baseline.py --arch vit_base_resnet50_384 --size 384 --lr 1e-3 --bs 24 --epochs 24