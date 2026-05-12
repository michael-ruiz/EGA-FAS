import argparse
import os
import time
import math
import torch
import numpy as np
from torch.utils.benchmark import timer
from torch import optim

from loss.optimizer import get_optimizer
from loss.adjust_lr import get_lr_scheduler
from loss.loss_fun import *
from loss.metric import *
from utils_other import get_save_path
from data_process.get_dataset import bulid_dataset
from model.bulid_model import *
from config.config import config
from loss.Count_params import count_params


# 用来对保存的模型命名，不同时间运行的程序文件名不同
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def run_train(localtime):
    save_path, file_name = get_save_path(config, localtime)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 模型保存路径设置
    if not os.path.exists(save_path + '/checkpoint'):
        os.makedirs(save_path + '/checkpoint')

    log_file = open(save_path+'/' + file_name, 'w')

    # get datasets
    train_loader, val_loader, test_loader = bulid_dataset(config)

    # get net
    net = get_model(config=config, num_class=2)
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net)
        net = net.cuda()
    else:
        net = net.to(device)

    # 参数量和FLOPs计算,并存储
    count_params(net, config, log_file, input_size=config.image_size)

    if config.pretrained_model is not None:
        print("loader pretrained model")
        init_checkpoint = os.path.join(save_path +'/checkpoint', config.pretrained_model)
        net.load_state_dict(torch.load(init_checkpoint, map_location=lambda storage, loc: storage))

    # criterion = softmax_cross_entropy_criterion
    label_smoothing = getattr(config, 'label_smoothing', 0.0)
    criterion = get_criterion(device=device, class_num=2, label_smoothing=label_smoothing)

    # weight_decay = 0.00005
    optimizer = get_optimizer(net, config)
    sgdr = get_lr_scheduler(config, optimizer, net)
    ls_str = f", label_smoothing:{config.label_smoothing}" if getattr(config, 'label_smoothing', 0) > 0 else ""
    aug_str = ", strong_augment" if getattr(config, 'strong_augment', False) else ""
    Model_description = f"Dataset:{config.dataset_name}, Protocol:{config.prot}_{config.sub_prot}\n" \
                        f"model:{config.model}, epochs:{config.epochs}, batchsize:{config.batch_size}, lr:{config.lr}\n" \
                        f"LOSS--BCE:{config.bce}, PWL:{config.pwl}, CMFL:{config.cmfl}, CCE:{config.cce}, ML:{config.ml}{ls_str}\n" \
                        f"Augmentation: {'strong' if getattr(config, 'strong_augment', False) else 'normal'}{aug_str}\n"
    print(Model_description)
    log_file.write(Model_description)
    log_file.flush()
    ##################### start training here! ##########################
    print("\n ************** start training here! **************")
    log_file.write('** start training here! **\n')
    train_f = '                                  |------------ VALID -------------|-------- TRAIN/BATCH ----------|      \n' \
              'model_name   lr   iter  epoch     |     loss      acer      acc    |     loss              acc     |  time\n' \
              '----------------------------------------------------------------------------------------------------\n'
    log_file.write(train_f)
    log_file.flush()
    print(train_f)

    val_min_acer = 1.0
    test_min_acer = 1.0
    Loss, ACER, ACC = 0, 0, 0
    start = timer()
    if config.dataset_name in ['CASIA-RA', 'RA-CASIA', 'CASIA_FASD-RA', 'RA-CASIA_FASD', 'WMCA-HQWMCA', 'HQWMCA-WMCA']:
        print_seq = 100  # eval from epoch 0 for cross-dataset (epochs // 100 ≈ 0)
    else:
        print_seq = 2
    for epoch in range(config.epochs):
        batch_loss, lr = train_one_epoch(epoch, train_loader, net, criterion, sgdr, optimizer, config)

        # Anneal Gumbel-Softmax temperature: linear decay from gumbel_tau to 0.1
        model_ref = net.module if hasattr(net, 'module') else net
        if hasattr(model_ref, 'use_gumbel') and model_ref.use_gumbel:
            tau = max(0.1, config.gumbel_tau * (1 - epoch / config.epochs))
            model_ref.set_gumbel_tau(tau)

        if epoch >= config.epochs // print_seq:
        # if True:
            net.eval()

            if config.dataset_name in ['CASIA-RA', 'RA-CASIA', 'CASIA_FASD-RA', 'RA-CASIA_FASD', 'WMCA-HQWMCA', 'HQWMCA-WMCA']:
                # 验证集测试
                val_loss, val_eval, val_TPR_FPRS, val_min_acer = test_write_reslt(epoch, net, val_loader, criterion, log_file, save_path, localtime, val_min_acer, state='val')
                Loss, ACER, ACC = val_loss, val_eval["ACER"], val_eval["ACC"]
                # 测试集测试 (always evaluate cross-domain; don't wait for val ACER=0)
                test_loss, test_eval, test_TPR_FPRS, test_min_acer = test_write_reslt(epoch, net, test_loader, criterion, log_file, save_path, localtime, test_min_acer, state='test')
                    
            else:
                test_loss, test_eval, test_TPR_FPRS, test_min_acer = test_write_reslt(epoch, net, val_loader, criterion, log_file, save_path, localtime, test_min_acer, state='test')
                Loss, ACER, ACC = test_loss, test_eval["ACER"], test_eval["ACC"]

        asterisk = ' '
        train_log = config.flod_name + ' epoch %d: %0.4f  %6.1f | %0.6f  %0.6f  %0.5f %s  | %0.6f  %0.6f |%s \n' % (
            epoch, lr, epoch, Loss, ACER, ACC, asterisk,  batch_loss[0], batch_loss[1],
            time_to_str((timer() - start), 'min'))
        log_file.write(train_log)
        log_file.flush()
        print(train_log,end='')


def test_write_reslt(epoch, net, test_loader, criterion, log_file, save_path, localtime, test_min_acer, state='test'):
    test_loss, test_correct, test_probs, test_labels = do_test(net, test_loader, criterion, config)
    test_eval, test_TPR_FPRS, threshold = model_performances(test_probs, test_labels)
    test_line = "{}: ERR: {:.6f}, ACER: {:.6f}, APCER: {:.6f}, NPCER: {:.6f}, ACC: {:.6f}, threshold: {:.3f}, " \
                "TPR@FPR=10E-2: {:.6f}, TPR@FPR=10E-3: {:.6f}, TPR@FPR=10E-4: {:.6f}\n".format(state,
        test_eval["ERR"], test_eval["ACER"], test_eval["APCER"], test_eval["BPCER"], test_eval["ACC"],threshold,
        test_TPR_FPRS["TPR@FPR=10E-2"], test_TPR_FPRS["TPR@FPR=10E-3"], test_TPR_FPRS["TPR@FPR=10E-4"])
    log_file.write(test_line)
    log_file.flush()

    if test_eval["ACER"] < test_min_acer and epoch > 0:
        test_min_acer = test_eval["ACER"]
        test_ckpt_name = save_path + f'/checkpoint/{state}_min_acer_model_' + localtime + '.pth'
        torch.save(net.state_dict(), test_ckpt_name)
        print(f'save min {state} acer model: {test_min_acer}')
        log_file.write(f'save min {state} acer model: ' + str(test_min_acer) + '\n')
        log_file.flush()

    return test_loss, test_eval, test_TPR_FPRS, test_min_acer


def run_test(config,localtime):
    save_path, _ = get_save_path(config, localtime)
    init_checkpoint = config.pretrained_model
    is_pruning = True
        
    if init_checkpoint is None:
        all_init_checkpoint = []
        all_file = os.listdir(save_path + '/checkpoint')
        for file in all_file:
            if file.endswith('.pth'):
                all_init_checkpoint.append(file)
    else:
        all_init_checkpoint = []
        all_init_checkpoint.append(config.pretrained_model)
    for one_checkpoint in all_init_checkpoint:
        init_checkpoint = os.path.join(save_path + '/checkpoint', one_checkpoint)
        print('\t loader initial_checkpoint = %s\n' % init_checkpoint)
        # get datasets
        train_loader, val_loader, test_loader = bulid_dataset(config)
        label_smoothing = getattr(config, 'label_smoothing', 0.0)
        criterion = get_criterion(device=device, class_num=2, label_smoothing=label_smoothing)

        print('infer!!!!!!!!!')
        # get net
        net = get_model(config=config, num_class=2)
        # Load checkpoint and handle DataParallel prefix
        state_dict = torch.load(init_checkpoint, map_location=lambda storage, loc: storage)
        # Remove 'module.' prefix if model was saved with DataParallel
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        try:
            net.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f'  Skipping incompatible checkpoint: {e}')
            continue
        if torch.cuda.is_available():
            net = net.cuda()
        else:
            net = net.to(device)
        net.eval()
        test_loss, test_correct, test_probs, test_labels = do_test(net, test_loader, criterion, config)     
        
        test_eval, test_TPR_FPRS, threshold = model_performances(test_probs, test_labels)
        test_line = "Test: ERR: {:.6f},  APCER: {:.6f}, NPCER: {:.6f}, ACER: {:.6f}, ACC: {:.6f}\n" \
                    "TPR@FPR=10E-2: {:.6f}, TPR@FPR=10E-3: {:.6f}, TPR@FPR=10E-4: {:.6f}\n".format(
            test_eval["ERR"], test_eval["APCER"], test_eval["BPCER"],test_eval["ACER"], test_eval["ACC"],
            test_TPR_FPRS["TPR@FPR=10E-2"], test_TPR_FPRS["TPR@FPR=10E-3"], test_TPR_FPRS["TPR@FPR=10E-4"])
        print(test_line, 'end')
        print('done!')

        file_name = save_path + '/checkpoint/' + one_checkpoint + '_test.txt'
        with open(file_name, 'w') as fm:
            # valid_loader上的测试结果
            # fm.write(line)
            # test_loader上的测试结果
            fm.write(test_line)            


def main():
    localtime = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    if config.mode == 'train':
        run_train(localtime)

    if config.mode == 'infer_test':
        config.pretrained_model = None  # "test_min_acer_model_20230119_05_54_21.pth"
        run_test(config,localtime)

    return


if __name__ == '__main__':
    print(config)
    from train_test.train_test_amplification import train_one_epoch, do_test
    if config.is_Multi:
        from train_test.train_test_Multi import train_one_epoch, do_test
    for i in range(config.cycle):
        main()
    config.mode = 'infer_test'
    main()