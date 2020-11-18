import os, argparse
import torch, pyro
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from utils.net import SSVAE
from pyro.contrib.examples.util import print_and_log
from torch.utils.data import DataLoader
from utils.datasets import TrainDataset, ValidDataset, TestDataset
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam
from tensorboardX import SummaryWriter

def run_inference_for_epoch(sup_data_loader, unsup_data_loader, loss_fn, use_cuda, ratio):
    sup_batches = len(sup_data_loader) #4
    unsup_batches = len(unsup_data_loader) #5
    batches_per_epoch = sup_batches + unsup_batches

    sup_iter = iter(sup_data_loader)
    unsup_iter = iter(unsup_data_loader)
    epoch_losses_sup = 0.
    epoch_losses_unsup = 0.

    for i in range(batches_per_epoch):
        ctr_sup = 0
        is_unsupervised = (i % 2 == 0) and (ctr_sup < sup_batches)
        if is_unsupervised:
            (xs, ys) = next(unsup_iter)
            if use_cuda:
                xs = xs.cuda()
            new_loss = loss_fn.step(xs)
            epoch_losses_unsup += new_loss
        else:
            (xs, ys) = next(sup_iter)
            ctr_sup += 1
            if use_cuda:
                xs = xs.cuda()
                ys = ys.cuda()
            new_loss = loss_fn.step(xs, ys, ratio)
            epoch_losses_sup += new_loss

    return epoch_losses_sup, epoch_losses_unsup

def run_inference_for_epoch_ncls(sup_data_loader, unsup_data_loader, loss_fn, use_cuda):
    sup_batches = len(sup_data_loader) #4
    unsup_batches = len(unsup_data_loader) #5
    batches_per_epoch = sup_batches + unsup_batches

    sup_iter = iter(sup_data_loader)
    unsup_iter = iter(unsup_data_loader)
    epoch_losses_sup = 0.
    epoch_losses_unsup = 0.

    for i in range(batches_per_epoch):
        ctr_sup = 0
        is_unsupervised = (i % 2 == 0) and (ctr_sup < sup_batches)
        if is_unsupervised:
            (xs, ys) = next(unsup_iter)
            if use_cuda:
                xs = xs.cuda()
            new_loss = loss_fn.step(xs)
            epoch_losses_unsup += new_loss
        else:
            (xs, ys) = next(sup_iter)
            ctr_sup += 1
            if use_cuda:
                xs = xs.cuda()
                ys = ys.cuda()
            new_loss = loss_fn.step(xs, ys)
            epoch_losses_sup += new_loss

    return epoch_losses_sup, epoch_losses_unsup

def compute_epoch_losses(data_loader, loss_fn, use_cuda, sup_flag):
    epoch_losses = 0.
    for xs, ys in data_loader:
        if use_cuda:
            xs = xs.cuda()
            ys = ys.cuda()
        if sup_flag:
            new_loss = loss_fn.evaluate_loss(xs, ys)
        else:
            new_loss = loss_fn.evaluate_loss(xs)
        epoch_losses += new_loss
    return epoch_losses

def get_evaluate_for_epoch(train_dataloader, test_dataloader, loss_basic, loss_aux, use_cuda):
    epoch_losses_sup = compute_epoch_losses(train_dataloader, loss_basic, use_cuda, sup_flag=True)
    epoch_losses_aux = compute_epoch_losses(train_dataloader, loss_aux, use_cuda, sup_flag=True)
    epoch_losses_unsup = compute_epoch_losses(test_dataloader, loss_basic, use_cuda, sup_flag=False)
    return epoch_losses_sup, epoch_losses_aux, epoch_losses_unsup

def get_accuracy_and_kappa(dataset, classifier):
    xs, ys = dataset.trials, dataset.labels
    y_pred = classifier(xs)
    _, y_pred = torch.max(y_pred, 1)
    acc = accuracy_score(ys.data.cpu().numpy(), y_pred.data.cpu().numpy())
    kappa = cohen_kappa_score(ys.data.cpu().numpy(), y_pred.data.cpu().numpy())
    return acc, kappa

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# session = 'A05'
parser = argparse.ArgumentParser()
parser.add_argument('--session', nargs='?', default='A01', type=str)
parser.add_argument('--factor', nargs='?', default=10.0, type=float)
args = parser.parse_args()
session = args.session
factor = args.factor
print('factor:{0:.3f}.'.format(factor))

# sessions = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']

# for session in sessions:
use_cuda = True
# learning_rate = 0.001
# beta_1 = 0.9
num_particles = 51
num_epochs = 150

pyro.set_rng_seed(1234)
ssvae = SSVAE(use_cuda=True)

# adam_params = {"lr": 0.00042, "betas": (0.9, 0.999), "weight_decay": 5e-3}
adam_params = {"lr": 0.001, "betas": (0.9, 0.999), "weight_decay": 5e-3}
optimizer = Adam(adam_params)
guide = config_enumerate(ssvae.guide, default="parallel", expand=True)
elbo = TraceEnum_ELBO(max_plate_nesting=1, num_particles=num_particles, vectorize_particles=True)
loss = SVI(ssvae.model, guide, optimizer, loss=elbo)

guide_ncls = config_enumerate(ssvae.guide_ncls, default="parallel", expand=True)
elbo_basic = TraceEnum_ELBO(max_plate_nesting=1, num_particles=num_particles, vectorize_particles=True)
loss_basic = SVI(ssvae.model_ncls, guide_ncls, optimizer, loss=elbo_basic)
elbo_aux = Trace_ELBO()
loss_aux = SVI(ssvae.model_classify, ssvae.guide_classify, optimizer, loss=elbo_aux)

modelfile = './ssvae_model/{0}/'.format(session)
if not os.path.exists(modelfile):
    os.makedirs(modelfile)

logfile = './logfile/{0}.txt'.format(session)
if not os.path.exists(os.path.dirname(logfile)):
    os.makedirs(os.path.dirname(logfile))
logger = open(logfile, 'w') if logfile else None

log_dir = './log_summary_writer/{0}/'.format(session)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)

train_dataset = TrainDataset(session=session)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)

valid_dataset = ValidDataset(session=session)

test_dataset = TestDataset(session=session)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

best_valid_kappa, cor_test_kappa, cor_epoch_1 = 0.0, 0.0, 0
best_test_kappa, cor_valid_kappa, cor_epoch_2 = 0.0, 0.0, 0

for epoch in range(num_epochs):

    ssvae.eval()
    epoch_losses_sup, epoch_losses_aux, epoch_losses_unsup = get_evaluate_for_epoch(train_dataloader,
                                                                                    test_dataloader,
                                                                                    loss_basic,
                                                                                    loss_aux,
                                                                                    use_cuda)
    epoch_losses_sup = epoch_losses_sup/len(train_dataset)
    epoch_losses_aux = epoch_losses_aux/len(train_dataset)
    epoch_losses_unsup = epoch_losses_unsup/len(test_dataset)
    ratio = np.abs(epoch_losses_sup/epoch_losses_aux)
    str_print = 'Epoch {0} (eval mode): epoch_losses_sup, {1:.3f}; epoch_losses_aux, {2:.3f}; ' \
                'epoch_losses_unsup, {3:.3f}; ratio, {4:.4f}'\
        .format(epoch, epoch_losses_sup, epoch_losses_aux, epoch_losses_unsup, ratio)
    print_and_log(logger, str_print)

    writer.add_scalar('Eval mode/epoch_losses_sup', epoch_losses_sup, epoch)
    writer.add_scalar('Eval mode/epoch_losses_aux', epoch_losses_aux, epoch)
    writer.add_scalar('Eval mode/epoch_losses_unsup', epoch_losses_unsup, epoch)
    writer.add_scalar('Eval mode/ratio', ratio, epoch)

    '''Train'''
    ssvae.train()
    if epoch_losses_aux<0.0012 and ratio>400000:
        epoch_losses_sup, epoch_losses_unsup = run_inference_for_epoch_ncls(train_dataloader, test_dataloader,
                                                                       loss_basic, use_cuda)
    # elif ratio>10000:
    #     epoch_losses_sup, epoch_losses_unsup = run_inference_for_epoch(train_dataloader, test_dataloader,
    #                                                                    loss, use_cuda, ratio * factor)
    else:
        epoch_losses_sup, epoch_losses_unsup = run_inference_for_epoch(train_dataloader,test_dataloader,
                                                                       loss,use_cuda, ratio*factor)
    epoch_losses_sup = epoch_losses_sup/len(train_dataset)
    epoch_losses_unsup = epoch_losses_unsup/len(test_dataset)
    str_print = 'Epoch {0}: epoch_losses_sup, {1:.3f}; epoch_losses_unsup, {2:.3f}.'\
        .format(epoch, epoch_losses_sup, epoch_losses_unsup)
    print_and_log(logger, str_print)

    if np.isnan(epoch_losses_sup) or np.isnan(epoch_losses_unsup):
        break

    writer.add_scalar('Train mode/epoch_losses_sup', epoch_losses_sup, epoch)
    writer.add_scalar('Train mode/epoch_losses_unsup', epoch_losses_unsup, epoch)

    ssvae.eval()
    xs = train_dataset.trials
    ys = torch.topk(train_dataset.labels, 1)[1].squeeze()
    if use_cuda:
        xs = xs.cuda()
        ys = ys.cuda()
    outputs = ssvae.classifier(xs)
    _, y_pred = torch.max(outputs, 1)
    train_acc = accuracy_score(ys.data.cpu().numpy(), y_pred.data.cpu().numpy())
    train_kappa = cohen_kappa_score(ys.data.cpu().numpy(), y_pred.data.cpu().numpy())

    '''Valid'''
    xs = valid_dataset.trials
    ys = torch.topk(valid_dataset.labels, 1)[1].squeeze()
    if use_cuda:
        xs = xs.cuda()
        ys = ys.cuda()
    outputs = ssvae.classifier(xs)
    _, y_pred = torch.max(outputs, 1)
    valid_acc = accuracy_score(ys.data.cpu().numpy(), y_pred.data.cpu().numpy())
    valid_kappa = cohen_kappa_score(ys.data.cpu().numpy(), y_pred.data.cpu().numpy())

    '''Test'''
    xs = test_dataset.trials
    ys = torch.topk(test_dataset.labels, 1)[1].squeeze()
    if use_cuda:
        xs = xs.cuda()
        ys = ys.cuda()
    outputs = ssvae.classifier(xs)
    _, y_pred = torch.max(outputs, 1)
    test_acc = accuracy_score(ys.data.cpu().numpy(), y_pred.data.cpu().numpy())
    test_kappa = cohen_kappa_score(ys.data.cpu().numpy(), y_pred.data.cpu().numpy())

    str_print = 'Epoch {0}: train_acc, {1:.3f}; valid_acc, {2:.3f}; test_acc, {3:.3f}' \
        .format(epoch, train_acc, valid_acc, test_acc)
    print_and_log(logger, str_print)

    str_print = 'Epoch {0}: train_kappa, {1:.3f}; valid_kappa, {2:.3f}; test_kappa, {3:.3f}' \
        .format(epoch, train_kappa, valid_kappa, test_kappa)
    print_and_log(logger, str_print)

    writer.add_scalar('Accuracy/train_acc', train_acc, epoch)
    writer.add_scalar('Accuracy/valid_acc', valid_acc, epoch)
    writer.add_scalar('Accuracy/test_acc', test_acc, epoch)

    writer.add_scalar('Kappa/train_kappa', train_kappa, epoch)
    writer.add_scalar('Kappa/valid_kappa', valid_kappa, epoch)
    writer.add_scalar('Kappa/test_kappa', test_kappa, epoch)

    if test_kappa > best_test_kappa:
        best_test_kappa = test_kappa
        cor_valid_kappa = valid_kappa
        cor_epoch_2 = epoch
        torch.save(ssvae.state_dict(), modelfile + 'ssvae_tc.pth')

    if valid_kappa > best_valid_kappa:
        best_valid_kappa = valid_kappa
        cor_test_kappa = test_kappa
        cor_epoch_1 = epoch
        torch.save(ssvae.state_dict(), modelfile + 'ssvae_vc.pth')

str_print = 'Best Valid: Epoch,{0}; best_valid_kappa,{1:.3f}; cor_test_kappa,{2:.3f}' \
    .format(cor_epoch_1, best_valid_kappa, cor_test_kappa)
print_and_log(logger, str_print)

str_print = 'Best Test: Epoch,{0}; best_test_kappa,{1:.3f}; cor_valid_kappa,{2:.3f}' \
    .format(cor_epoch_2, best_test_kappa, cor_valid_kappa)
print_and_log(logger, str_print)

writer.close()
logger.close()
print('Good!')