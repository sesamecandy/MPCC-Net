# Experiment all tricks without center loss without re-ranking: 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on (=raw all trick, softmax_triplet.yml)
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# without center loss
# without re-ranking
python3 tools/test.py \
--config_file='configs/softmax_triplet.yml' \
MODEL.DEVICE_ID "('5')" \
MODEL.NAME "('resnet50_ibn_a')" \
DATASETS.NAMES "('market1501')"  \
DATASETS.ROOT_DIR "('/home/weidong.shi1/data')" \
MODEL.PRETRAIN_CHOICE "('self')" \
OUTPUT_DIR "('./results/FSCAC/Market')"