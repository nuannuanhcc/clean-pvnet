python run.py --type evaluate --cfg_file configs/linemod.yaml \
 train.dataset CocoTrain \
 test.dataset CocoTest \
 gpus [1,] \
 model mycat

# python run.py --type evaluate --cfg_file configs/linemod.yaml model cat