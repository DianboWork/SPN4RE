## NYT Partial Match
```shell
python -m main --num_generated_triplets 15 --na_rel_coef 1 --max_grad_norm 1 --max_epoch 100 --visible_gpu 4 --max_span_length 10
```

## NYT Exact Match
```shell
nohup python -m main --num_generated_triplets 10 --max_grad_norm 1 --na_rel_coef 1 --max_epoch 100 --max_span_length 10
```
result: 92.07

```shell
python -m main --num_generated_triplets 9 --max_grad_norm 1 --na_rel_coef 0.25 --max_epoch 100 --max_span_length 10
```

result: 92.08

```shell
python -m main --num_generated_triplets 15 --max_grad_norm 2.5 --na_rel_coef 0.25 --max_epoch 100 --visible_gpu 4 --max_span_length 10
```
result: 92.45
```shell
python -m main --num_generated_triplets 15 --max_grad_norm 1 --na_rel_coef 0.5 --max_epoch 100 --visible_gpu 3 --max_span_length 10
```
result: 92.43
## WebNLG Partial Match
```shell
python -m main --batch_size 4 --num_generated_triplets 10 --na_rel_coef 0.25 --max_grad_norm 20  --max_epoch 100 --visible_gpu 4 --encoder_lr 0.00002 --decoder_lr 0.00005 --num_decoder_layers 4 --max_span_length 10 --weight_decay 0.000001 --lr_decay 0.02
```

result: 93.4