import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import torch
from torch.autograd import Variable, grad
from torch.nn.functional import binary_cross_entropy_with_logits as bce

from tqdm import tqdm

import os

if not os.path.exists('./samples'):
    os.makedirs('./samples')

n_iterations = 20001
n_latent = 2
n_layers = 3
n_hidden = 512
bs = 128
extraD = 5
use_cuda = True

for shape in ['ring', 'grid']:
    for n_latent in [2, 5, 10]:
        for mixup in [0, 0.1, 0.2, 0.5, 1]:

            class Perceptron(torch.nn.Module):
                def __init__(self, sizes, final=None):
                    super(Perceptron, self).__init__()
                    layers = []
                    for i in range(len(sizes) - 1):
                        layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
                        if i != (len(sizes) - 2):
                            layers.append(torch.nn.ReLU(inplace=True))
                    if final is not None:
                        layers.append(final())
                    self.net = torch.nn.Sequential(*layers)

                def forward(self, x):
                    return self.net(x)

            def plot(x, y, mixup, iteration):
                lims = (x.min() - .25, x.max() + .25)
                plt.figure(figsize=(2, 2))
                plt.plot(x[:, 0], x[:, 1], '.', label='real')
                plt.plot(y[:, 0], y[:, 1], '.', alpha=0.25, label='fake')
                plt.axis('off')
                plt.gca().axes.get_xaxis().set_visible(False)
                plt.gca().axes.get_yaxis().set_visible(False)
                plt.xlim(*lims)
                plt.ylim(*lims)
                plt.tight_layout(0, 0, 0)
                plt.show()
                plt.savefig("images/example_z=%d_%s_%1.1f_%06d.png" %
                            (n_latent, shape, mixup, iteration),
                            bbox_inches='tight', pad_inches=0)
                plt.close()

            def means_circle(k=8):
                p = 3.14159265359
                t = torch.linspace(0, 2 * p - (2 * p / k), k)
                m = torch.cat((torch.sin(t).view(-1, 1),
                               torch.cos(t).view(-1, 1)), 1)
                return m

            def means_grid(k=25):
                m = torch.zeros(k, 2)
                s = int(torch.sqrt(torch.Tensor([k]))[0] / 2)
                cnt = 0
                for i in range(- s, s + 1):
                    for j in range(- s, s + 1):
                        m[cnt][0] = i
                        m[cnt][1] = j
                        cnt += 1
                return m / s

            def sample_real(n, shape, std=0.01):
                if shape == 'ring':
                    m = means_circle()
                else:
                    m = means_grid()
                i = torch.zeros(n).random_(m.size(0)).long()
                s = torch.randn(n, 2) * std + m[i]
                s = Variable(s, requires_grad=True)
                if use_cuda:
                    s = s.cuda()
                return s

            def sample_noise(bs, d):
                z = torch.randn(bs, d)
                z = Variable(z, requires_grad=True)
                if use_cuda:
                    z = z.cuda()
                return z

            netD = Perceptron([2] + [n_hidden] * n_layers + [1])
            netG = Perceptron([n_latent] + [n_hidden] * n_layers + [2])

            if use_cuda:
                netD.cuda()
                netG.cuda()

            optD = torch.optim.Adam(netD.parameters())
            optG = torch.optim.Adam(netG.parameters())

            p_real = sample_real(1000, shape)
            p_nois = sample_noise(1000, n_latent)

            def mixup_batch(mixup=0.0):
                def one_batch():
                    real = sample_real(bs, shape)
                    fake = netG(sample_noise(bs, n_latent))
                    data = torch.cat((real, fake))
                    ones = Variable(torch.ones(real.size(0), 1))
                    zeros = Variable(torch.zeros(fake.size(0), 1))
                    perm = torch.randperm(data.size(0)).view(-1).long()
                    if use_cuda:
                        ones = ones.cuda()
                        zeros = zeros.cuda()
                        perm = perm.cuda()
                    labels = torch.cat((ones, zeros))
                    return data[perm], labels[perm]

                d1, l1 = one_batch()
                if mixup == 0:
                    return d1, l1
                d2, l2 = one_batch()
                alpha = Variable(torch.randn(d1.size(0), 1).uniform_(0, mixup))
                if use_cuda:
                    alpha = alpha.cuda()
                d = alpha * d1 + (1. - alpha) * d2
                l = alpha * l1 + (1. - alpha) * l2
                return d, l

            for iteration in tqdm(range(n_iterations)):
                for extra in range(extraD):
                    data, labels = mixup_batch(mixup)

                    optD.zero_grad()
                    lossD = bce(netD(data), labels)
                    lossD.backward()
                    optD.step()

                data, labels = mixup_batch(0)

                optG.zero_grad()
                lossG = - bce(netD(data), labels)
                lossG.backward()
                optG.step()

                if iteration in [10, 100, 1000, 10000, 20000]:
                    plot_real = p_real.cpu().data.numpy()
                    plot_fake = netG(p_nois).cpu().data.numpy()
                    torch.save((plot_real, plot_fake),
                               'samples/example_z=%d_%s_%1.1f_%06d.pt' %
                               (n_latent, shape, mixup, iteration))
                    plot(plot_real, plot_fake, mixup, iteration)
