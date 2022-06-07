CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 -H localhost:2 python3 dreamerv2/train.py --logdir ~/logdir/atari_riverraid/horovod/1 --configs atari --task atari_riverraid
