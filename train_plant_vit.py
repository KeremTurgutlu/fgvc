from fastai.vision.all import *
from self_supervised.layers import *
from fastcore.script import *
from fastai.callback.wandb import WandbCallback
import wandb

from self_supervised.multimodal.clip import *
import clip
import gc
from vit import *
torch.backends.cudnn.benchmark = True

datapath = Path("plant-pathology-2021-fgvc8/")
train_df = pd.read_csv(datapath/'train.csv')
files = train_df['image'].values
fn2label = dict(zip(train_df['image'], train_df['labels'].apply(lambda o: o.split())))

def read_img(im): return PILImage.create(datapath/f'train_images_512/{im}')
def read_label(fn): return fn2label[fn]


def get_dls(files, bs=32, size=448):
    tfms = [[read_img, ToTensor, RandomResizedCrop(size, min_scale=.7)], 
            [read_label, MultiCategorize(), OneHotEncode]]
    dsets = Datasets(files, tfms=tfms, splits=RandomSplitter(valid_pct=0.1, seed=0.2)(files))
    batch_augs = aug_transforms()
    batch_tfms = [IntToFloatTensor] + batch_augs
    dls = dsets.dataloaders(bs=bs, after_batch=batch_tfms)
    return dls

def create_vitb_model(size, patch_size=32, patch_stride=32, clip_pretrained=True, checkpoint=False, checkpoint_nchunks=2):
    "ViT-B model with 12 layers, 12 heads, width 768, embed size 512 and custom patch size and stride size"
    if clip_pretrained:
        if patch_size != 32 and patch_stride != 32: raise Exception(f"Patch size and stride needs to be 32 for pretrained model")
        print("Loading pretrained model..")
        vitb32_config_dict = vitb32_config(224, context_length=77, vocab_size=49408)
        clip_model = CLIP(**vitb32_config_dict, checkpoint_nchunks=checkpoint_nchunks)
        clip_pretrained_model, _ = clip.load("ViT-B/32", jit=False)
        clip_model.load_state_dict(clip_pretrained_model.state_dict())

        clip_vitb = clip_model.visual
        num_patches = (size//patch_size)**2 +1
        # interpolate positional embedding to match any input size
        embed_dim = clip_vitb.positional_embedding.size(1)
        clip_vitb.positional_embedding.data = F.interpolate(clip_vitb.positional_embedding.data[None, None, ...], 
                                                            size=[num_patches, embed_dim], 
                                                            mode='bilinear',align_corners=False)[0,0]
        del clip_model, clip_pretrained_model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        clip_vitb = VisualTransformer(size,patch_size=patch_size,patch_stride=patch_stride,
                                      width=768,layers=12,heads=12,output_dim=512,
                                      checkpoint=checkpoint, checkpoint_nchunks=checkpoint_nchunks)
    
    return clip_vitb

@patch
def name(self:F1ScoreMulti): return "f1_multi_macro"
f1macro = F1ScoreMulti(average='macro')
@patch
def name(self:F1ScoreMulti): return "f1_multi_micro"
f1micro = F1ScoreMulti(average='micro')


@call_parse
def main(
    size:               Param("Size in px", int)=384,
    lr:                 Param("Learning rate", float)=1e-2,
    bs:                 Param("Batch Size", int)=32,
    patch_sizes:        Param("Patch Sizes", str)=32,
    patch_strides:      Param("Patch Strides", str)=32,
    clip_pretrained:    Param("CLIP pretrained", bool_arg)=False,
    checkpoint:         Param("Do grad checkpoint", bool_arg)=False,
    checkpoint_nchunks: Param("Num chunks for grad checkpoint", int)=2,
    epochs:             Param("Num epochs", int)=24):
    
    "ViT training script for https://www.kaggle.com/c/plant-pathology-2021-fgvc8/overview"
    
    patch_sizes, patch_strides = eval(patch_sizes), eval(patch_strides)
    print(patch_sizes, patch_strides)
    
    WANDB = True
    if WANDB:
        xtra_config = {"Arch":"ViT-B/X", "Size":size,"Patch Sizes":patch_sizes, 
                       "Patch Strides":patch_strides,  "lr": lr}
        wandb.init(project="plant-pathology-2021", config=xtra_config)
    
    dls = get_dls(files,bs,size)
    encoder = VisualTransformer(size,patch_sizes=patch_sizes, patch_strides=patch_strides,
                                      width=768,layers=12,heads=12,output_dim=512,
                                      checkpoint=checkpoint, checkpoint_nchunks=checkpoint_nchunks)
    print(encoder.convs)
    with torch.no_grad(): nf = encoder(torch.randn(2,3,size,size)).size(-1)
    classifier = create_cls_module(nf, dls.c)
    model = nn.Sequential(encoder, classifier)
    cbs = [SaveModelCallback(every_epoch=True, 
                             fname=f"vitb-{patch_sizes}-{patch_strides}_size{size}_lr{lr}_epochs{epochs}")]
    if WANDB: cbs += [WandbCallback(log_preds=False,log_model=False)]
    learn = Learner(dls, model, opt_func=ranger, cbs=cbs,
                    metrics=[f1macro, f1micro], loss_func=BCEWithLogitsLossFlat())
    learn.to_fp16()
#     learn.fit_flat_cos(epochs,lr,pct_start=0.25)
    learn.fit_one_cycle(epochs,lr,pct_start=0.25,div=1e3,div_final=1e5)
    if WANDB: wandb.finish()