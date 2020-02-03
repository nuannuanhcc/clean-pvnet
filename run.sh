python train_net.py --cfg_file configs/linemod.yaml \
 train.dataset CocoTrain \
 test.dataset CocoTest \
 train.batch_size 32 \
 gpus [1,] \
 model mycat

#python train_net.py --cfg_file configs/linemod.yaml model mycat cls_type cat gpus [1,]