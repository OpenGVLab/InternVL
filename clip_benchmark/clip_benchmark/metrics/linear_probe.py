import os
import time
from contextlib import suppress

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .zeroshot_classification import accuracy


def assign_learning_rate(param_group, new_lr):
    param_group['lr'] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


class Featurizer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        # note: not sure if we want to train on l2-normalized features
        image_features = self.model.encode_image(input)
        image_features = F.normalize(image_features, dim=-1)
        return image_features


class FeatureDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i], self.targets[i]


def evaluate(model, train_dataloader, dataloader, fewshot_k, batch_size, num_workers, lr, epochs,
             model_id, seed, feature_root, device, amp=True, verbose=False):
    # warning: we currently only support non-multi-label classification datasets.
    assert device == 'cuda'  # need to use cuda for this else too slow
    # first we need to featurize the dataset, and store the result in feature_root
    if not os.path.exists(feature_root):
        os.mkdir(feature_root)
    feature_dir = os.path.join(feature_root, model_id)
    if not os.path.exists(feature_dir):
        os.mkdir(feature_dir)

    featurizer = Featurizer(model).cuda()
    autocast = torch.cuda.amp.autocast if amp else suppress
    if not os.path.exists(os.path.join(feature_dir, 'targets_train.pt')):
        # now we have to cache the features
        devices = [x for x in range(torch.cuda.device_count())]
        featurizer = torch.nn.DataParallel(featurizer, device_ids=devices)

        for j, loader in enumerate([dataloader, train_dataloader]):
            save_str = '_train' if j == 1 else '_val'
            features = []
            targets = []
            num_batches_tracked = 0
            num_cached = 0
            with torch.no_grad():
                for images, target in tqdm(loader):
                    images = images.to(device)

                    with autocast():
                        feature = featurizer(images)

                    features.append(feature.cpu())
                    targets.append(target)

                    num_batches_tracked += 1
                    if (num_batches_tracked % 100) == 0:
                        features = torch.cat(features)
                        targets = torch.cat(targets)

                        torch.save(features, os.path.join(feature_dir, f'features{save_str}_cache_{num_cached}.pt'))
                        torch.save(targets, os.path.join(feature_dir, f'targets{save_str}_cache_{num_cached}.pt'))
                        num_cached += 1
                        features = []
                        targets = []

            if len(features) > 0:
                features = torch.cat(features)
                targets = torch.cat(targets)
                torch.save(features, os.path.join(feature_dir, f'features{save_str}_cache_{num_cached}.pt'))
                torch.save(targets, os.path.join(feature_dir, f'targets{save_str}_cache_{num_cached}.pt'))
                num_cached += 1

            features = torch.load(os.path.join(feature_dir, f'features{save_str}_cache_0.pt'))
            targets = torch.load(os.path.join(feature_dir, f'targets{save_str}_cache_0.pt'))
            for k in range(1, num_cached):
                next_features = torch.load(os.path.join(feature_dir, f'features{save_str}_cache_{k}.pt'))
                next_targets = torch.load(os.path.join(feature_dir, f'targets{save_str}_cache_{k}.pt'))
                features = torch.cat((features, next_features))
                targets = torch.cat((targets, next_targets))

            for k in range(num_cached):
                os.remove(os.path.join(feature_dir, f'features{save_str}_cache_{k}.pt'))
                os.remove(os.path.join(feature_dir, f'targets{save_str}_cache_{k}.pt'))

            torch.save(features, os.path.join(feature_dir, f'features{save_str}.pt'))
            torch.save(targets, os.path.join(feature_dir, f'targets{save_str}.pt'))

    features = torch.load(os.path.join(feature_dir, 'features_train.pt'))
    targets = torch.load(os.path.join(feature_dir, 'targets_train.pt'))

    # second, make a dataloader with k features per class. if k = -1, use all features.
    length = len(features)
    perm = [p.item() for p in torch.randperm(length)]
    idxs = []
    counts = {}
    num_classes = 0

    for p in perm:
        target = targets[p].item()
        if target not in counts:
            counts[target] = 0
            num_classes += 1

        if fewshot_k < 0 or counts[target] < fewshot_k:
            counts[target] += 1
            idxs.append(p)

    for c in counts:
        if fewshot_k > 0 and counts[c] != fewshot_k:
            print('insufficient data for this eval')
            return

    features = features[idxs]
    targets = targets[idxs]
    feature_dset = FeatureDataset(features, targets)

    # now train the model
    feature_loader = DataLoader(feature_dset, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers,
                                pin_memory=True,
                                )

    probe = torch.nn.Linear(features[0].shape[0], targets.max().item() + 1)
    devices = [x for x in range(torch.cuda.device_count())]
    probe = probe.cuda()
    probe = torch.nn.DataParallel(probe, device_ids=devices)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=lr,
        weight_decay=0,
    )
    criterion = torch.nn.CrossEntropyLoss()

    len_loader = len(feature_loader)
    scheduler = cosine_lr(optimizer, lr, 0., epochs * len_loader)

    for epoch in range(epochs):
        end = time.time()
        for i, (x, y) in enumerate(feature_loader):
            x, y = x.cuda(), y.cuda()
            step = i + epoch * len_loader
            scheduler(step)
            data_time = time.time() - end

            optimizer.zero_grad()
            with autocast():
                pred = probe(x)
                loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()

            if (i % 20) == 0:
                num_samples = i * len(x)
                try:
                    samples_per_epoch = len(train_dataloader)
                    percent_complete = 100.0 * i / len(train_dataloader)
                    progress_message = f'[{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]'
                except TypeError:
                    progress_message = f'[{num_samples} samples]'
                print(
                    f'Train Epoch: {epoch} {progress_message}\t'
                    f'Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}\t'
                    f"LR {optimizer.param_groups[0]['lr']:.5f}"
                )

    # finally, evaluate.
    features = torch.load(os.path.join(feature_dir, 'features_val.pt'))
    targets = torch.load(os.path.join(feature_dir, 'targets_val.pt'))
    feature_dset = FeatureDataset(features, targets)
    feature_loader = DataLoader(feature_dset, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers,
                                pin_memory=True,
                                )
    true, pred = [], []
    with torch.no_grad():
        for x, y in tqdm(feature_loader):
            x = x.to(device)
            y = y.to(device)

            with autocast():
                # predict
                logits = probe(x)

            pred.append(logits.cpu())
            true.append(y.cpu())

    logits = torch.cat(pred)
    target = torch.cat(true)
    pred = logits.argmax(axis=1)

    # measure accuracy
    if target.max() >= 5:
        acc1, acc5 = accuracy(logits.float(), target.float(), topk=(1, 5))
    else:
        acc1, = accuracy(logits.float(), target.float(), topk=(1,))
        acc5 = float('nan')
    mean_per_class_recall = balanced_accuracy_score(target, pred)
    if verbose:
        print(classification_report(target, pred, digits=3))

    print('acc1:', acc1)
    return {'lp_acc1': acc1, 'lp_acc5': acc5, 'lp_mean_per_class_recall': mean_per_class_recall,
            'lr': lr, 'epochs': epochs, 'seed': seed, 'fewshot_k': fewshot_k}
