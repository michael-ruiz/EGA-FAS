import torch
import torch.nn.functional as F
import numpy as np
from loss.metric import *
from tqdm import tqdm

# Device configuration (CPU or CUDA)
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def train_one_epoch(epoch, train_loader, net, criterion, sgdr, optimizer, config):
    sgdr.step()
    lr = optimizer.param_groups[0]['lr']
    print('epoch-{:}: lr={:.4f}'.format(epoch, lr))
    batch_loss = np.zeros(6, np.float32)
    r1, r2 = config.bce, config.pwl
    r3 = config.cmfl
    optimizer.zero_grad()

    # Track predictions for debugging
    all_predictions = []
    all_truths = []

    for input, mask, truth in train_loader:
        # one iteration update
        net.train()
        input = input.to(device)
        truth = truth.to(device)
        mask = mask.to(device)
        outputs = net.forward(input)
        logit, depth_logits, color_logits, ir_logits, x_map = outputs[:5]
        aux_logits = outputs[6] if len(outputs) > 6 else None
        truth = truth.view(logit.shape[0])

        x_map = x_map.to(torch.float32)
        mask = mask.to(torch.float32)

        loss_pwl = criterion['PWL'](x_map, mask)
        loss1 = criterion['BCE'](logit, truth)
        loss_r,loss_d,loss_i,loss2 = criterion['ICMFL'](color_logits, depth_logits, ir_logits, truth)
        loss = r1*loss1 + r2*loss_pwl + r3 * loss2

        # Per-branch auxiliary loss (forces all Gumbel branches to be competent)
        if aux_logits is not None and hasattr(config, 'aux_loss_weight') and config.aux_loss_weight > 0:
            aux_loss = sum(F.cross_entropy(a, truth) for a in aux_logits) / len(aux_logits)
            loss = loss + config.aux_loss_weight * aux_loss

        # MoE-style load balancing loss
        guidance_weights = outputs[5] if len(outputs) > 5 else None
        if guidance_weights is not None and hasattr(config, 'balance_loss_weight') and config.balance_loss_weight > 0:
            eps = 1e-8
            # Confidence: minimize per-sample entropy → sharper per-sample weights
            per_sample_entropy = -(guidance_weights * torch.log(guidance_weights + eps)).sum(dim=1).mean()
            # Balance: maximize batch-average entropy → prevent collapse to one modality
            batch_avg = guidance_weights.mean(dim=0)
            batch_entropy = -(batch_avg * torch.log(batch_avg + eps)).sum()
            # Combined: want low per-sample entropy + high batch entropy
            balance_loss = per_sample_entropy - batch_entropy
            loss = loss + config.balance_loss_weight * balance_loss

        # Safety check: stop if NaN detected
        if torch.isnan(loss):
            print(f"ERROR: NaN loss detected at epoch {epoch}!")
            print(f"  loss1 (BCE): {loss1.item()}, loss_pwl: {loss_pwl.item()}, loss2 (ICMFL): {loss2.item()}")
            import sys
            sys.exit(1)
        # loss = loss1 + r3 * loss2
        # logit = (depth_logits + color_logits + ir_logits) / 3.0

        precision, prob = metric(logit, truth)

        # Debug: Track predictions
        if epoch == 0:
            pred_class = torch.argmax(prob, dim=1).cpu().numpy()
            all_predictions.extend(pred_class)
            all_truths.extend(truth.cpu().numpy())

        loss.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)

        optimizer.step()
        optimizer.zero_grad()
        batch_loss[:2] = np.array((loss.item(), precision.item(),))  # 损失函数和准确率 [2.6929493 0.671875 ]

    # Debug: Print training prediction distribution for epoch 0
    if epoch == 0:
        all_predictions = np.array(all_predictions)
        all_truths = np.array(all_truths)
        print(f"[Training Debug] Total samples: {len(all_predictions)}, "
              f"Predicted class 0: {np.sum(all_predictions == 0)}, "
              f"Predicted class 1: {np.sum(all_predictions == 1)}")
        print(f"[Training Debug] True labels - Class 0: {np.sum(all_truths == 0)}, "
              f"Class 1: {np.sum(all_truths == 1)}")
        print(f"[Training Debug] Accuracy: {np.mean(all_predictions == all_truths):.4f}")

    return batch_loss, lr

def do_test(net, test_loader, criterion, config):
    valid_num = 0
    losses = []
    corrects = []
    probs = []
    probs1, probs2, probs3, probs4 = [], [], [], []
    labels = []
    all_guidance_weights = []  # Collect guidance weights for analysis
    # for i, (input, truth) in enumerate(tqdm(test_loader)):
    for i, (input, mask, truth) in enumerate(test_loader):
        b, n, c, w, h = input.size()
        input = input.view(b * n, c, w, h)
        input = input.to(device)
        truth = truth.to(device)
        with torch.no_grad():
            outputs = net(input)
            logit, depth_logits, color_logits, ir_logits, x_map = outputs[:5]
            guidance_weights = outputs[5] if len(outputs) > 5 else None
            if guidance_weights is not None:
                # Average weights over n augmentations per sample: [b*n, M] -> [b, n, M] -> [b, M]
                num_mod = guidance_weights.shape[1]
                gw = guidance_weights.view(b, n, num_mod).mean(dim=1)
                all_guidance_weights.append(gw.cpu())
            logit = logit.view(b, n, logit.shape[1])

            # color_logits = color_logits.view(b, n, color_logits.shape[1])
            # depth_logits = depth_logits.view(b, n, depth_logits.shape[1])
            # ir_logits = ir_logits.view(b, n, ir_logits.shape[1])

            logit = torch.mean(logit, dim=1, keepdim=False)

            # color_logits = torch.mean(color_logits, dim=1, keepdim=False)
            # depth_logits = torch.mean(depth_logits, dim=1, keepdim=False)
            # ir_logits = torch.mean(ir_logits, dim=1, keepdim=False)

            # logit = (depth_logits + color_logits + ir_logits) / 3.0

            truth = truth.view(b)
            loss = criterion['BCE'](logit, truth)
            correct, prob = metric(logit, truth)
            # correct1, prob1 = metric(depth_logits, truth)
            # correct2, prob2 = metric(color_logits, truth)
            # correct3, prob3 = metric(ir_logits, truth)
            

        valid_num += len(input)
        losses.append(loss.data.cpu().numpy())
        corrects.append(np.asarray(correct).reshape([1]))
        probs.append(prob.data.cpu().numpy())
        labels.append(truth.data.cpu().numpy())

        # probs1.append(prob1.data.cpu().numpy())
        # probs2.append(prob2.data.cpu().numpy())
        # probs3.append(prob3.data.cpu().numpy())
        

    correct = np.concatenate(corrects)
    correct = np.mean(correct)
    loss = np.array(losses)
    loss = loss.mean()
    probs = np.concatenate(probs)

    # Debug: Check prediction distribution
    predictions = np.argmax(probs, axis=1)
    print(f"[Validation Debug] Total samples: {len(predictions)}, "
          f"Predicted class 0: {np.sum(predictions == 0)}, "
          f"Predicted class 1: {np.sum(predictions == 1)}")
    print(f"[Validation Debug] Prob stats - Min: {probs.min():.4f}, Max: {probs.max():.4f}, "
          f"Mean: {probs.mean():.4f}")

    probs = probs[:, 1]
    labels = np.concatenate(labels)

    # Debug: Check label distribution
    print(f"[Validation Debug] True labels - Class 0: {np.sum(labels == 0)}, "
          f"Class 1: {np.sum(labels == 1)}")


    """
    probs1 = np.concatenate(probs1)
    probs1 = probs1[:, 1].T
    test_eval1, test_TPR_FPRS1, threshold1 = model_performances(probs1, labels)

    probs2 = np.concatenate(probs2)
    probs2 = probs2[:, 1].T
    test_eval_rgb, test_TPR_FPRS_rgb, threshold_rgb = model_performances(probs2, labels)

    probs3 = np.concatenate(probs3)
    probs3 = probs3[:, 1].T
    test_eval_ir, test_TPR_FPRS_ir, threshold_ir = model_performances(probs3, labels)


    print(f"Depth分支===ACC: {test_eval1['ACC']}, ACER: {test_eval1['ACER']}, APCER: {test_eval1['APCER']}, BPCER: {test_eval1['BPCER']}\n"
          f"color分支===ACC: {test_eval_rgb['ACC']}, ACER: {test_eval_rgb['ACER']}, APCER: {test_eval_rgb['APCER']}, BPCER: {test_eval_rgb['BPCER']}\n"
          f"ir分支===   ACC: {test_eval_ir['ACC']}, ACER: {test_eval_ir['ACER']}, APCER: {test_eval_ir['APCER']}, BPCER: {test_eval_ir['BPCER']}\n"
          )
    """

    # Guidance weights analysis (for adaptive guidance mode)
    if all_guidance_weights:
        weights = torch.cat(all_guidance_weights, dim=0)  # [N, M]
        num_mod = weights.shape[1]
        mod_names = ['color', 'depth', 'ir', 'thermal'][:num_mod]
        print(f"\n{'='*50}")
        print(f"=== Guidance Weights Analysis ===")
        print(f"{'='*50}")
        print(f"Mean weights: " + ", ".join(f"{n}={weights[:, i].mean():.4f}" for i, n in enumerate(mod_names)))
        print(f"Std weights:  " + ", ".join(f"{n}={weights[:, i].std():.4f}" for i, n in enumerate(mod_names)))
        print(f"Min weights:  " + ", ".join(f"{n}={weights[:, i].min():.4f}" for i, n in enumerate(mod_names)))
        print(f"Max weights:  " + ", ".join(f"{n}={weights[:, i].max():.4f}" for i, n in enumerate(mod_names)))

        # Per-class analysis (real vs spoof)
        labels_tensor = torch.from_numpy(labels)
        real_mask = (labels_tensor == 0)
        spoof_mask = (labels_tensor == 1)

        if real_mask.sum() > 0:
            print(f"\nReal samples ({real_mask.sum().item()}):  " +
                  ", ".join(f"{n}={weights[real_mask, i].mean():.4f}" for i, n in enumerate(mod_names)))
        if spoof_mask.sum() > 0:
            print(f"Spoof samples ({spoof_mask.sum().item()}): " +
                  ", ".join(f"{n}={weights[spoof_mask, i].mean():.4f}" for i, n in enumerate(mod_names)))
        print(f"{'='*50}\n")

    return loss, correct, probs, labels