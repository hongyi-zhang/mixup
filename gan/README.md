
# mixup training of GAN
This is a demo implementation of using _mixup_ in GAN training of two 2-d toy examples as shown in the [paper](https://arxiv.org/abs/1710.09412).

## Training
Simply run
```
python example_gan.py
```

You will need PyTorch and the `tqdm` package to run this script.

## Results
It may take a few hours (about 5 hours on a Nvidia GTX 1070) to run all the settings in this script. After the experiments finish, you should see a set of images similar to the ones shown here in the `images` folder, which are visualizations of the target distribution and generated samples during the training process. The `samples` folder contains the `(x, y)` values of corresponding samples.

![](images/gan_results.png)
