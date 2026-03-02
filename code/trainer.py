import torch
import logging
import time
import numpy as np
from sklearn import metrics
import torch.nn.functional as F
import torch.optim as optim
from run_manager import RunManager
from torch.utils.data import DataLoader


# =============================================================================
# Training loop
# =============================================================================

def train(model, train_set, dev_set, test_set, hyper_params, batch_size, device):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    m = RunManager()
    optimizer = optim.AdamW(model.parameters(), lr=hyper_params.learning_rate, weight_decay=1e-4)
    # Cosine annealing LR schedule: helps transformers converge consistently
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hyper_params.num_epoch, eta_min=hyper_params.learning_rate * 0.01
    )

    num_batches = len(train_loader)
    model_name = model.__class__.__name__

    header = (
        f"\n{'='*70}\n"
        f"  Model        : {model_name}\n"
        f"  LR           : {hyper_params.learning_rate}\n"
        f"  Epochs       : {hyper_params.num_epoch}\n"
        f"  Batch size   : {batch_size}\n"
        f"  Train batches: {num_batches}\n"
        f"{'='*70}"
    )
    print(header)
    logging.info(header)

    m.begin_run(hyper_params, model, train_loader)

    for epoch in range(hyper_params.num_epoch):
        m.begin_epoch(epoch + 1)
        model.train()

        epoch_loss = 0.0
        t0 = time.time()
        current_lr = scheduler.get_last_lr()[0] if epoch > 0 else hyper_params.learning_rate

        for batch_idx, batch in enumerate(train_loader):
            texts   = batch['text'].to(device)
            targets = batch['codes'].to(device)

            outputs, ldam_outputs, _ = model(texts, targets)

            if ldam_outputs is not None:
                loss = F.binary_cross_entropy_with_logits(ldam_outputs, targets)
            else:
                loss = F.binary_cross_entropy_with_logits(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping: prevents exploding gradients in deep transformers
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            m.track_loss(loss)

            # ---- Print batch progress every 50 batches ----
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
                elapsed = time.time() - t0
                avg_loss = epoch_loss / (batch_idx + 1)
                print(
                    f"  Epoch [{epoch+1:3d}/{hyper_params.num_epoch}] "
                    f"Batch [{batch_idx+1:4d}/{num_batches}] "
                    f"| Loss: {loss.item():.4f} "
                    f"| Avg Loss: {avg_loss:.4f} "
                    f"| LR: {current_lr:.2e}"
                    f"| Time: {elapsed:.1f}s",
                    flush=True
                )

        scheduler.step()

        # ---- Epoch summary ----
        epoch_avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - t0
        new_lr = scheduler.get_last_lr()[0]
        summary = (
            f"\n  ─── Epoch {epoch+1:3d}/{hyper_params.num_epoch} Done "
            f"| Avg Train Loss: {epoch_avg_loss:.4f} "
            f"| LR: {new_lr:.2e} "
            f"| Time: {epoch_time:.1f}s ───"
        )
        print(summary, flush=True)
        logging.info(f"Epoch {epoch+1} | avg_loss={epoch_avg_loss:.4f} | lr={new_lr:.2e} | time={epoch_time:.1f}s")

        m.end_epoch()

    m.end_run()
    hype = '_'.join([f'{k}_{v}' for k, v in hyper_params._asdict().items()])
    m.save(f'../results/train_results_{hype}')
    logging.info("Training finished.\n")
    print(f"\n{'='*70}\n  Training complete.\n{'='*70}\n")

    # ---- Evaluation ----
    print("  Evaluating on TRAIN set...")
    train_loader_eval = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=1)
    probabs, tgts, _, _ = evaluate(model, train_loader_eval, device, dtset='train')
    compute_scores(probabs, tgts, hyper_params, dtset='train')

    print("\n  Evaluating on DEV set...")
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=1)
    probabs, tgts, _, _ = evaluate(model, dev_loader, device, dtset='dev')
    compute_scores(probabs, tgts, hyper_params, dtset='dev')

    print("\n  Evaluating on TEST set...")
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)
    probabs, tgts, full_hadm_ids, full_attn_weights = evaluate(model, test_loader, device, dtset='test')
    compute_scores(probabs, tgts, hyper_params, dtset='test',
                   full_hadm_ids=full_hadm_ids, full_attn_weights=full_attn_weights)


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, loader, device, dtset):
    fin_targets = []
    fin_probabs = []
    full_hadm_ids = []
    full_attn_weights = []

    with torch.no_grad():
        model.eval()
        for batch in loader:
            hadm_ids = batch['hadm_id']
            texts    = batch['text'].to(device)
            targets  = batch['codes']

            outputs, _, attn_weights = model(texts)

            fin_targets.extend(targets.tolist())
            fin_probabs.extend(torch.sigmoid(outputs).detach().cpu().tolist())
            if dtset == 'test' and attn_weights is not None:
                full_hadm_ids.extend(hadm_ids)
                full_attn_weights.extend(attn_weights.detach().cpu().tolist())

    return fin_probabs, fin_targets, full_hadm_ids, full_attn_weights


# =============================================================================
# Metrics
# =============================================================================

def save_predictions(probabs, targets, dtset, hype):
    np.savetxt(f'../results/{dtset}_probabs_{hype}.txt', probabs)
    np.savetxt(f'../results/{dtset}_targets_{hype}.txt', targets)


def precision_at_k(true_labels, pred_probs):
    ks = [1, 5, 8, 10, 15]
    sorted_pred = np.argsort(pred_probs)[:, ::-1]
    output = []
    p5_scores = None
    for k in ks:
        topk = sorted_pred[:, :k]
        vals = []
        for i, tk in enumerate(topk):
            if len(tk) > 0:
                num_true_in_top_k = true_labels[i, tk].sum()
                vals.append(num_true_in_top_k / float(len(tk)))
        output.append(np.mean(vals))
        if k == 5:
            p5_scores = np.array(vals)
    return output, p5_scores


def compute_scores(probabs, targets, hyper_params, dtset,
                   full_hadm_ids=None, full_attn_weights=None):
    probabs = np.array(probabs)
    targets = np.array(targets)

    preds           = np.rint(probabs)
    accuracy        = metrics.accuracy_score(targets, preds)
    f1_micro        = metrics.f1_score(targets, preds, average='micro')
    f1_macro        = metrics.f1_score(targets, preds, average='macro')
    auc_micro       = metrics.roc_auc_score(targets, probabs, average='micro')
    auc_macro       = metrics.roc_auc_score(targets, probabs, average='macro')
    precision_at_ks, p5_scores = precision_at_k(targets, probabs)

    result_block = (
        f"\n  ┌─ [{dtset.upper()}] Results ─────────────────────────────────────┐\n"
        f"  │  Accuracy       : {accuracy:.4f}\n"
        f"  │  F1  (micro)    : {f1_micro:.4f}\n"
        f"  │  F1  (macro)    : {f1_macro:.4f}\n"
        f"  │  AUC (micro)    : {auc_micro:.4f}\n"
        f"  │  AUC (macro)    : {auc_macro:.4f}\n"
        f"  │  P@k [1,5,8,10,15]: {[f'{v:.4f}' for v in precision_at_ks]}\n"
        f"  └────────────────────────────────────────────────────────────┘"
    )
    print(result_block, flush=True)
    logging.info(result_block)

    if dtset == 'test' and full_attn_weights:
        hype = '_'.join([f'{k}_{v}' for k, v in hyper_params._asdict().items()])
        save_predictions(probabs, targets, dtset, hype)
        full_attn_weights = np.array(full_attn_weights)
        sorted_idx   = np.argsort(p5_scores)[::-1]
        sorted_pred  = np.argsort(probabs)[:, ::-1]
        top5_preds   = sorted_pred[:, :5]
        with open(f'../results/{dtset}_attn_weights_{hype}.txt', 'w') as fout:
            for idx in sorted_idx:
                fout.write(f'idx: {idx}\n')
                fout.write(f'{full_hadm_ids[idx]};{p5_scores[idx]}\n')
                fout.write(f'{sorted_pred[idx]}\n')
                fout.write(f'{targets[idx, sorted_pred[idx]]}\n')
                fout.write(f'{preds[idx, sorted_pred[idx]]}\n')
                fout.write(f'{probabs[idx, sorted_pred[idx]]}\n')
                weights = full_attn_weights[idx, top5_preds[idx]]
                for wlist in weights:
                    fout.write(' '.join([str(val) for val in wlist]) + '\n')
