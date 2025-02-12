# !/bin/bash -ex

for seed in 0 1 2 3 4
do
  python -m semiSupervised.simgrace_meta \
    --seed $seed \
    --device 0 \
    --model sparse_cin \
    --use_coboundaries True \
    --nonlinearity relu \
    --jump_mode cat \
    --graph_norm bn \
    --lr 0.001 \
    --num_layers 3 \
    --emb_dim 32 \
    --batch_size 128 \
    --dataset NCI1 \
    --max_dim 2 \
    --max_ring_size 6 \
    --exp_name 0131-gumbel-semi \
    --init_method mean \
    --eta 0.1
done