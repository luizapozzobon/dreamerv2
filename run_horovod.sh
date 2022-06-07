CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 -H localhost:2 python dreamerv2/train.py --logdir /logdir/atari_riverraid/dreamerv2/teste --configs atari --task atari_riverraid
