This repo contains a demo reimplementation (since the original code for the paper belongs to Facebook) of the CIFAR-10 training code in PyTorch based on the following paper:
> Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin and David Lopez-Paz. _mixup: Beyond Empirical Risk Minimization._ https://arxiv.org/abs/1710.09412

## Training
```bash
# training with default parameters (weight_decay=1e-4 and alpha=1)
python easy_mixup.py --sess my_session_1 --seed 11111
```
The above bash script will download the CIFAR data and train the network with 1e-4 weight decay and mixup parameter alpha=1.0; alternatively, we can experiment with other weight decay and alpha value using corresponding options:
```bash
# training with weight_decay=5e-4 and alpha=0 (no mixup)
python easy_mixup.py --sess my_session_2 --seed 22222 --decay 5e-4 --alpha 0.
```
The other choices (network architecture, #epochs, learning rate schedule, momentum, data augmentation etc.) are hard coded but modifications are hopefully straightfoward.

By default, the trained model with the best validation accuracy resides in `./checkpoint` folder, and the training log (including training loss/accuracy and validation loss/accuracy for each epoch) is saved in `./results` as a `.csv` file.

## Other frameworks
 - [A Tensorflow implementation of mixup](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet#cifar10-preact18-mixuppy) which reproduces our results in [tensorpack](https://github.com/ppwwyyxx/tensorpack)

## Acknowledgement
This reimplementation is adapted from the [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) repository by [kuangliu](https://github.com/kuangliu).
