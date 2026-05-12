import torch
import numpy as np
from loss.metric import *
from tqdm import tqdm

def train_one_epoch(epoch, train_loader, net, criterion, sgdr, optimizer, config):
    sgdr.step()
    lr = optimizer.param_groups[0]['lr']
    print('epoch-{:}: lr={:.4f}'.format(epoch, lr))
    batch_loss = np.zeros(6, np.float32)

    optimizer.zero_grad()
    gama = config.bce

    for input, mask, truth in train_loader:
        # one iteration update
        net.train()
        input = input.cuda()
        truth = truth.cuda()
        mask = mask.cuda()
        logit, x_map = net.forward(input)
        truth = truth.view(logit.shape[0])
        loss = criterion['BCE'](logit, truth)

        x_map = x_map.to(torch.float32)
        mask = mask.to(torch.float32)
        loss1 = criterion['PWL'](x_map, mask)
        loss = gama*loss + (1-gama)*loss1

        precision, _ = metric(logit, truth)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        optimizer.step()
        optimizer.zero_grad()
        batch_loss[:2] = np.array((loss.item(), precision.item(),))  # 损失函数和准确率 [2.6929493 0.671875 ]

    return batch_loss, lr

def do_test(net, test_loader, criterion, config):
    valid_num = 0
    losses = []
    corrects = []
    probs = []
    labels = []
    # for i, (input, truth) in enumerate(tqdm(test_loader)):
    for i, (input, mask, truth) in enumerate(test_loader):
        b, n, c, w, h = input.size()
        input = input.view(b * n, c, w, h)
        input = input.cuda()
        truth = truth.cuda()
        with torch.no_grad():
            logit, x_map = net(input)
            logit = logit.view(b, n, logit.shape[1])
            logit = torch.mean(logit, dim=1, keepdim=False)
            truth = truth.view(b)
            # truth = truth.repeat(n)
            # truth = torch.repeat_interleave(truth, n, dim=0)
            loss = criterion['BCE'](logit, truth)
            correct, prob = metric(logit, truth)

        valid_num += len(input)
        losses.append(loss.data.cpu().numpy())
        corrects.append(np.asarray(correct).reshape([1]))
        probs.append(prob.data.cpu().numpy())
        labels.append(truth.data.cpu().numpy())
    correct = np.concatenate(corrects)
    correct = np.mean(correct)
    loss = np.array(losses)
    loss = loss.mean()
    probs = np.concatenate(probs)
    probs = probs[:, 1]
    labels = np.concatenate(labels)

    return loss, correct, probs, labels