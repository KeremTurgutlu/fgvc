from fastai.vision.all import *
from self_supervised.layers import *
from fastcore.script import *
from fastai.callback.wandb import WandbCallback
import wandb
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

@patch
def name(self:F1ScoreMulti): return "f1_multi_macro"
f1macro = F1ScoreMulti(average='macro')
@patch
def name(self:F1ScoreMulti): return "f1_multi_micro"
f1micro = F1ScoreMulti(average='micro')


@call_parse
def main(
    arch:   Param("Architecture", str)='tf_efficientnet_b4_ns',
    size:   Param("Size (px: 128,192,256)", int)=448,
    lr:     Param("Learning rate", float)=1e-2,
    bs:     Param("Size (px: 128,192,256)", int)=32,
    epochs: Param("Size (px: 128,192,256)", int)=24):
    
    "Baseline training script for https://www.kaggle.com/c/plant-pathology-2021-fgvc8/overview"

    
    WANDB = True
    if WANDB:
        xtra_config = {"Arch":arch, "Size":size, "lr": lr}
        wandb.init(project="plant-pathology-2021", config=xtra_config)
    
    dls = get_dls(files,bs,size)
    encoder = create_encoder(arch, pretrained=False, n_in=3)
    if arch == 'vit_base_patch16_384':  encoder = CheckpointVisionTransformer(encoder, 2)
    if arch == 'vit_base_resnet50_384': encoder = CheckpointVisionTransformer(encoder, 12)
    with torch.no_grad(): nf = encoder(torch.randn(2,3,size,size)).size(-1)
    classifier = create_cls_module(nf, dls.c)
    model = nn.Sequential(encoder, classifier)
    cbs = [SaveModelCallback(every_epoch=True, fname=f"{arch}_size{size}_lr{lr}_bs_{bs}_epochs{epochs}")]
    if WANDB: cbs += [WandbCallback(log_preds=False,log_model=False)]
    learn = Learner(dls, model, opt_func=ranger, cbs=cbs,
                    metrics=[f1macro, f1micro], loss_func=BCEWithLogitsLossFlat())
    learn.to_fp16()
    learn.fit_flat_cos(epochs,lr,pct_start=0.25)
    if WANDB: wandb.finish()