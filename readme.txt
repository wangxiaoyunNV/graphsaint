Notes:
1. "modules", "sampler", "train_sampling", and "utils.py" are implemented by DGL. https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsaint
2. "my_sampler1" and "my_train_sampling1" are implemented to integrate cuGraph with DGL: using cuGraph RW to sample subgraphs and train GCN model using DGL.
3. "my_sampler2" and "my_train_sampling2" are used to run deeper RW and use walk paths as subgraphs instead of using subgraphs induced from sampled nodes by RW.

To run the code:
1. python train_sampling.py --gpu 0 --dataset ppi --sampler rw --num-roots 3000 --length 2 --num-repeat 50 --n-epochs 1000 --n-hidden 512 --arch 1-0-1-0 --dropout 0.1
2. python my_train_sampling1.py --gpu 0 --dataset ppi --sampler cugraph_rw --num-roots 3000 --length 2 --num-repeat 50 --n-epochs 1000 --n-hidden 512 --arch 1-0-1-0 --dropout 0.1
3. python my_train_sampling2.py --gpu 0 --dataset ppi --sampler cugraph_rw --num-roots 1000 --length 20 --num-repeat 50 --n-epochs 1000 --n-hidden 512 --arch 1-0-1-0 --dropout 0.1
4. python my_train_sampling3.py --gpu 0 --dataset ppi --sampler cugraph_ego --num-roots 3000 --length 2 --num-repeat 50 --n-epochs 1000 --n-hidden 512 --arch 1-0-1-0 --dropout 0.1



