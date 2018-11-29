python3 train_dist.py --embed_dim 128 \
                      --ff_embed_dim 256 \
                      --num_heads 8 \
                      --layers 2 \
                      --dropout 0.1 \
                      --train_data toy/train\
                      --vocab toy/vocab\
                      --batch_size 128\
                      --max_len 512 \
                      --world_size 2\
                      --gpus 2\
                      --MASTER_ADDR 100.88.66.72\
                      --MASTER_PORT 29500\
                      --start_rank 0\
                      --print_every 100\
                      --save_every 10000
