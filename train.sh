# Default
# For small models
# bash distributed_train.sh 8 --model repghost.repghostnet_0_5x -b 128 --lr 0.6 --sched cosine --epochs 300 --opt sgd -j 7 --warmup-epochs 5 --warmup-lr 1e-4 --weight-decay 1e-5 --drop 0.2 --amp --model-ema --model-ema-decay 0.9999 --remode pixel --reprob 0.2 --output work_dirs/train/
# bash distributed_train.sh 8 --model repghost.repghostnet_0_58x -b 128 --lr 0.6 --sched cosine --epochs 300 --opt sgd -j 7 --warmup-epochs 5 --warmup-lr 1e-4 --weight-decay 1e-5 --drop 0.2 --amp --model-ema --model-ema-decay 0.9999 --remode pixel --reprob 0.2 --output work_dirs/train/
# bash distributed_train.sh 8 --model repghost.repghostnet_0_8x -b 128 --lr 0.6 --sched cosine --epochs 300 --opt sgd -j 7 --warmup-epochs 5 --warmup-lr 1e-4 --weight-decay 1e-5 --drop 0.2 --amp --model-ema --model-ema-decay 0.9999 --remode pixel --reprob 0.2 --output work_dirs/train/
# bash distributed_train.sh 8 --model repghost.repghostnet_1_0x -b 128 --lr 0.6 --sched cosine --epochs 300 --opt sgd -j 7 --warmup-epochs 5 --warmup-lr 1e-4 --weight-decay 1e-5 --drop 0.2 --amp --model-ema --model-ema-decay 0.9999 --remode pixel --reprob 0.2 --output work_dirs/train/
# bash distributed_train.sh 8 --model repghost.repghostnet_1_11x -b 128 --lr 0.6 --sched cosine --epochs 300 --opt sgd -j 7 --warmup-epochs 5 --warmup-lr 1e-4 --weight-decay 1e-5 --drop 0.2 --amp --model-ema --model-ema-decay 0.9999 --remode pixel --reprob 0.2 --output work_dirs/train/


# For large models
# bash distributed_train.sh 8 --model repghost.repghostnet_1_3x -b 128 --lr 0.6 --sched cosine --epochs 300 --opt sgd -j 7 --warmup-epochs 5 --warmup-lr 1e-4 --weight-decay 1e-5 --drop 0.2 --amp --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --output work_dirs/train/
# bash distributed_train.sh 8 --model repghost.repghostnet_1_5x -b 128 --lr 0.6 --sched cosine --epochs 300 --opt sgd -j 7 --warmup-epochs 5 --warmup-lr 1e-4 --weight-decay 1e-5 --drop 0.2 --amp --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --output work_dirs/train/
# bash distributed_train.sh 8 --model repghost.repghostnet_2_0x -b 128 --lr 0.6 --sched cosine --epochs 300 --opt sgd -j 7 --warmup-epochs 5 --warmup-lr 1e-4 --weight-decay 1e-5 --drop 0.2 --amp --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --output work_dirs/train/

