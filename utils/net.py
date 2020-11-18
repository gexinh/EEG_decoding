import torch, pyro
import torch.nn as nn
import pyro.distributions as dist
from pyro.distributions.util import broadcast_shape

class ConcatModule(nn.Module):
    """
    a custom module for concatenation of tensors
    """
    def __init__(self, allow_broadcast=False):
        self.allow_broadcast = allow_broadcast
        super(ConcatModule, self).__init__()

    def forward(self, *input_args):
        # we have a single object
        if len(input_args) == 1:
            input_args = input_args[0]

        if torch.is_tensor(input_args):
            return input_args
        else:
            if self.allow_broadcast:
                shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
                input_args = [s.expand(shape) for s in input_args]
            return torch.cat(input_args, dim=-1)

class Encoder_Z1(nn.Module):
    def __init__(self, use_cuda=False):
        super(Encoder_Z1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 7), stride=(1, 3))
        self.relu_1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 3), padding=(0, 1))
        self.relu_2 = nn.ReLU()
        self.conv31 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(32, 1), groups=32)
        self.conv32 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(32, 1), groups=32)
        self.softplus = nn.Softplus()
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()

    def forward(self, x): #[N, 1, 32, 70]
        x = self.conv1(x)  # [N, 16, 32, 22]
        x = self.relu_1(x)
        x = self.conv2(x)  # [N, 32, 32, 8]
        x = self.relu_2(x)
        z1_mean = self.conv31(x)  # [N, 32, 1, 8]
        z1_mean = torch.flatten(z1_mean, start_dim=1, end_dim=-1) #[N, 256]
        z1_std = self.conv32(x) # [N, 32, 1, 8]
        z1_std = torch.flatten(z1_std, start_dim=1, end_dim=-1) #[N, 256]
        z1_std = self.softplus(z1_std)
        return z1_mean, z1_std

class Encoder_Z2(nn.Module):
    def __init__(self, use_cuda=False):
        super(Encoder_Z2, self).__init__()
        self.concat = ConcatModule(allow_broadcast=True)
        self.fc11 = nn.Linear(260, 50)
        self.fc12 = nn.Linear(260, 50)
        self.softplus = nn.Softplus()

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()

    def forward(self, z1, y):
        u = self.concat(z1, y)
        frontshape = list(u.size())[:-1]
        u = u.reshape([-1, ] + list(u.size())[-1:])

        z2_mean = self.fc11(u)
        z2_std = self.fc12(u)
        z2_std = self.softplus(z2_std)

        ns1 = frontshape + list(z2_mean.size())[-1:]
        z2_mean = z2_mean.reshape(ns1)
        z2_std = z2_std.reshape(ns1)

        return z2_mean, z2_std

class Encoder_Y(nn.Module):
    def __init__(self,  use_cuda=False):
        super(Encoder_Y, self).__init__()
        self.fc = nn.Linear(256, 4)
        self.softmax = nn.Softmax(dim=-1)

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()

    def forward(self, z1):
        frontshape = list(z1.size())[:-1]
        u = z1.reshape([-1, ] + list(z1.size())[-1:])

        u = self.fc(u)
        y = self.softmax(u)

        ns1 = frontshape + list(y.size())[-1:]
        y = y.reshape(ns1)
        return y

class Decoder_Z1(nn.Module):
    def __init__(self):
        super(Decoder_Z1, self).__init__()
        self.concat = ConcatModule(allow_broadcast=True)

        self.fc11 = nn.Linear(54, 256)
        self.fc12 = nn.Linear(54, 256)
        self.softplus = nn.Softplus()

    def forward(self, z2, y):
        u = self.concat(z2, y)
        frontshape = list(u.size())[:-1]
        u = u.reshape([-1, ] + list(u.size())[-1:])

        z1_mean = self.fc11(u)
        z1_std = self.fc12(u)
        z1_std = self.softplus(z1_std)

        ns1 = frontshape + list(z1_mean.size())[-1:]
        z1_mean = z1_mean.reshape(ns1)
        z1_std = z1_std.reshape(ns1)
        return z1_mean, z1_std

class Decoder_X(nn.Module):
    def __init__(self, use_cuda=False):
        super(Decoder_X, self).__init__()
        self.dconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(32, 1), groups=32)
        self.batchnormal_1 = nn.BatchNorm2d(num_features=32, momentum=0.1)
        self.relu_1 = nn.ReLU()
        self.dconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(1, 3), stride=(1, 3),
                                         padding=(0, 1))
        self.batchnormal_2 = nn.BatchNorm2d(num_features=16, momentum=0.1)
        self.relu_2 = nn.ReLU()
        self.dconv31 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1, 7), stride=(1, 3))
        self.softplus_1 = nn.Softplus(beta=1, threshold=20)
        self.dconv32 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1, 7), stride=(1, 3))
        self.softplus_2 = nn.Softplus(beta=1, threshold=20)

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()

    def forward(self, z1):
        frontshape = list(z1.size())[:-1]
        u = z1.reshape(frontshape + [32, 1, 8])
        u = u.reshape([-1, ] + [32, 1, 8])

        u = self.dconv1(u)
        u = self.batchnormal_1(u)
        u = self.relu_1(u)
        u = self.dconv2(u)
        u = self.batchnormal_2(u)
        u = self.relu_2(u)
        x_mean = self.dconv31(u)
        x_mean = self.softplus_1(x_mean)
        x_std = self.dconv32(u)
        x_std = self.softplus_2(x_std)

        ns1 = frontshape + list(x_mean.size())[-3:]
        x_mean = x_mean.reshape(ns1)
        x_std = x_std.reshape(ns1)

        return x_mean, x_std

class SSVAE(nn.Module):
    def __init__(self, aux_scale=46, use_cuda=False):
        super(SSVAE, self).__init__()
        self.aux_scale = aux_scale
        self.use_cuda = use_cuda
        self.encoder_z1 = Encoder_Z1()
        self.encoder_z2 = Encoder_Z2()
        self.encoder_y = Encoder_Y()
        self.decoder_z1 = Decoder_Z1()
        self.decoder_x = Decoder_X()
        if self.use_cuda:
            self.cuda()

    def model(self, xs, ys=None, aux_scale=46):
        pyro.module("ss_vae", self)
        batch_size = xs.size(0)
        options = dict(out=None, dtype=xs.dtype, layout=torch.strided, device=xs.device, requires_grad=False)

        with pyro.plate("data"):
            prior_loc = torch.zeros(batch_size, 50, **options)
            prior_scale = torch.ones(batch_size, 50, **options)
            zs2 = pyro.sample("z2", dist.Normal(prior_loc, prior_scale).to_event(1))

            alpha_prior = torch.ones(batch_size, 4, **options) / 4.0
            ys_ = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=ys)

            z1_mean, z1_std = self.decoder_z1(zs2, ys_)
            zs1 = pyro.sample("z1", dist.Normal(z1_mean, z1_std).to_event(1))
            x_mean, x_std = self.decoder_x(zs1)
            pyro.sample('x', dist.Normal(x_mean, x_std).to_event(3), obs=xs)

            if ys is not None:
                alpha = self.encoder_y(zs1)
                # with pyro.poutine.scale(scale=self.aux_scale):
                with pyro.poutine.scale(scale=aux_scale):
                    pyro.sample("y_aux", dist.OneHotCategorical(alpha), obs=ys)

    def guide(self, xs, ys=None, aux_scale=46):
        with pyro.plate("data"):
            z1_mean, z1_std = self.encoder_z1(xs)
            zs1 = pyro.sample("z1", dist.Normal(z1_mean, z1_std).to_event(1))

            if ys is None:
                alpha = self.encoder_y(zs1)
                ys = pyro.sample("y", dist.OneHotCategorical(alpha))

            z2_mean, z2_std = self.encoder_z2(zs1, ys)
            zs2 = pyro.sample("z2", dist.Normal(z2_mean, z2_std).to_event(1))
            pass

    def classifier(self, xs):
        zs1, _ = self.encoder_z1(xs)
        alpha = self.encoder_y(zs1)
        return alpha

    def model_classify(self, xs, ys=None):
        pyro.module("ss_vae", self)
        with pyro.plate("data"):
            if ys is not None:
                z1, _ = self.encoder_z1(xs)
                alpha = self.encoder_y(z1)
                pyro.sample("y_aux", dist.OneHotCategorical(alpha), obs=ys)

    def guide_classify(self, xs, ys=None):
        pass

    def model_ncls(self, xs, ys=None):
        pyro.module("ss_vae", self)
        batch_size = xs.size(0)
        options = dict(out=None, dtype=xs.dtype, layout=torch.strided, device=xs.device, requires_grad=False)

        with pyro.plate("data"):
            prior_loc = torch.zeros(batch_size, 50, **options)
            prior_scale = torch.ones(batch_size, 50, **options)
            zs2 = pyro.sample("z2", dist.Normal(prior_loc, prior_scale).to_event(1))

            alpha_prior = torch.ones(batch_size, 4, **options) / 4.0
            ys_ = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=ys)

            z1_mean, z1_std = self.decoder_z1(zs2, ys_)
            zs1 = pyro.sample("z1", dist.Normal(z1_mean, z1_std).to_event(1))
            x_mean, x_std = self.decoder_x(zs1)
            pyro.sample('x', dist.Normal(x_mean, x_std).to_event(3), obs=xs)

    def guide_ncls(self, xs, ys=None):
        with pyro.plate("data"):
            z1_mean, z1_std = self.encoder_z1(xs)
            zs1 = pyro.sample("z1", dist.Normal(z1_mean, z1_std).to_event(1))

            if ys is None:
                alpha = self.encoder_y(zs1)
                ys = pyro.sample("y", dist.OneHotCategorical(alpha))

            z2_mean, z2_std = self.encoder_z2(zs1, ys)
            zs2 = pyro.sample("z2", dist.Normal(z2_mean, z2_std).to_event(1))
            pass
