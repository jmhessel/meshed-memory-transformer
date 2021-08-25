'''
CUDA_VISIBLE_DEVICES=6 python train_clip.py --batch_size 50 --m 40 --head 8 --warmup 10000 --features_path coco_detections.hdf5 --annotation_folder annotations/ --workers 8 --exp_name CLIPScore_reinforce --resume_last --reward CLIPScore --use_for_early_stop CLIPScore --force_rl_start
'''
import random
from data import ImageDetectionsField, ImageDetectionsFieldWithID, TextField, RawField
from data import COCO, DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
import pprint
import subprocess
import json
import torch.nn.functional as F
import sklearn.preprocessing
import clip
import collections
import copy

_CLIP_DEVICE = 'cuda'
_CLIP_MODEL, _ =  clip.load('ViT-B/32', device=_CLIP_DEVICE, jit=False)
_CLIP_MODEL.eval()
_CLIP_IM_FEATS = np.load('all_mscoco_images_for_clip_extract/clip_features/trainval2017_image_features~ViT-B32.npy')
_CLIP_IM_FEATS = sklearn.preprocessing.normalize(_CLIP_IM_FEATS)
with open('all_mscoco_images_for_clip_extract/clip_features/trainval2017_image_features~ViT-B32~im2row.json') as f:
    _CLIP_IM2ROW = json.load(f)
    _CLIP_IM2ROW = {int(k): v for k, v in _CLIP_IM2ROW.items()}
print('loaded clip im features with {}-shape (id map len {})'.format(_CLIP_IM_FEATS.shape, len(_CLIP_IM2ROW)))
print(list(_CLIP_IM2ROW.keys())[:5])
random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def evaluate_loss(model, dataloader, loss_fn, text_field):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (detections, ids, captions) in enumerate(dataloader):
                detections, captions = detections.to(device), captions.to(device)
                out = model(detections, captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}

    all_gens, all_gts, all_ids = [], [], []
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, caps_gt) in enumerate(dataloader):
            images, ids = detections
            images = images.to(device)
            ids = ids.cpu().numpy().tolist()
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i, id_i) in enumerate(zip(caps_gt, caps_gen, ids)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
                all_gens.append(gen_i)
                all_gts.append(gts_i)
                all_ids.append(id_i)
            pbar.update()
            
    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    clipscore, refclipscore = compute_clipscore(all_gens, all_gts, all_ids, use_refclipscore=True)
    scores['CLIPScore'] = np.mean(clipscore)
    scores['RefCLIPScore'] = np.mean(refclipscore)
    scores['CIDErCLIPScore'] = (scores['CIDEr'] + scores['CLIPScore']) / 2
    return scores


def train_xe(model, dataloader, optim, text_field):
    # Training with cross-entropy
    model.train()
    scheduler.step()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, ids, captions) in enumerate(dataloader):
            detections, captions = detections.to(device), captions.to(device)
            out = model(detections, captions)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            scheduler.step()
            
    loss = running_loss / len(dataloader)
    return loss


# we will batch the mat
class MatDataset(torch.utils.data.Dataset):
    def __init__(self, mat):
        self.mat = mat
    def __len__(self):
        return len(self.mat)
    def __getitem__(self, idx):
        return self.mat[idx]


def batched_encode_text(toks):
    global _CLIP_MODEL, _CLIP_DEVICE
    iterable_mat = torch.utils.data.DataLoader(MatDataset(toks), batch_size=1024, shuffle=False)
    out = []
    with torch.no_grad():
        for batch in iterable_mat:
            out.append(_CLIP_MODEL.encode_text(batch))
        return torch.cat(out, dim=0)
    

def compute_clipscore(preds, refs, ids, use_refclipscore=False, w=2.5):
    global _CLIP_MODEL, _CLIP_IM_FEATS, _CLIP_IM2ROW, _CLIP_DEVICE

    flat_refs = []
    flat_refs_idxs = []
    for idx, c_refs in enumerate(refs):
        flat_refs.extend(c_refs)
        flat_refs_idxs.extend([idx for _ in c_refs])

    reflens = list(map(len, refs))

    flat_refs = []
    for r in refs:
        flat_refs.extend(r)

    im_feats = _CLIP_IM_FEATS[np.array([_CLIP_IM2ROW[im] for im in ids])]
    with torch.no_grad():
        toks = clip.tokenize(['A photo depicts ' + p for p in preds])
        toks = toks.to(_CLIP_DEVICE)
        preds_feats = F.normalize(batched_encode_text(toks)).cpu().numpy()

    clipscore = np.sum(im_feats * preds_feats, axis=1)
    clipscore = w*np.clip(clipscore, 0, None) # for the harmonic mean, and to prevent any (rare) negatives
    if not use_refclipscore: return clipscore
    with torch.no_grad():
        toks = clip.tokenize(['A photo depicts ' + p for p in flat_refs])
        toks = toks.to(_CLIP_DEVICE)
        flat_refs_feats = F.normalize(batched_encode_text(toks)).cpu().numpy()

    cand_idx2refs = collections.defaultdict(list)
    for ref_feats, cand_idx in zip(flat_refs_feats, flat_refs_idxs):
        cand_idx2refs[cand_idx].append(ref_feats)

    assert len(cand_idx2refs) == len(preds_feats)
    cand_idx2refs = {k: np.vstack(v) for k, v in cand_idx2refs.items()}
    per = []
    method = 'max'
    for c_idx, cand in enumerate(preds_feats):
        cur_refs = cand_idx2refs[c_idx]
        all_sims = cand.dot(cur_refs.transpose())
        if method == 'max':
            per.append(np.max(all_sims))
        elif method == 'mean':
            per.append(np.mean(all_sims))

    refonly_clipscore = np.array(per)
    refclipscore = 2 * clipscore * refonly_clipscore / (clipscore + refonly_clipscore)
    return clipscore, refclipscore


def train_scst(model, dataloader, optim, cider, text_field, reward_type):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    model.train()
    running_loss = .0
    seq_len = 20
    beam_size = 5

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, caps_gt) in enumerate(dataloader):
            detections, ids = detections
            detections = detections.to(device)
            outs, log_probs = model.beam_search(detections, seq_len, text_field.vocab.stoi['<eos>'],
                                                beam_size, out_size=beam_size)
            optim.zero_grad()
            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            ids = ids.cpu().numpy().tolist()
            ids = list(itertools.chain(*([c, ] * beam_size for c in ids)))
            if reward_type == 'CLIPScore':
                reward = compute_clipscore(caps_gen, caps_gt, ids)
            elif reward_type == 'RefCLIPScore':
                _, reward = compute_clipscore(caps_gen, caps_gt, ids, use_refclipscore=True)
            elif reward_type == 'CIDEr':
                caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
                reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            elif reward_type == 'CIDErCLIPScore':
                reward2 = compute_clipscore(caps_gen, caps_gt, ids)
                caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
                reward1 = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
                reward = (reward1 + reward2)/2
            else:
                raise NotImplementedError()
            reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--exp_name', type=str, default='m2_transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--use_for_early_stop', type=str, default='CLIPScore', choices=['CLIPScore', 'CIDEr', 'RefCLIPScore', 'CIDErCLIPScore'])
    parser.add_argument('--reward', type=str, default='CLIPScore', choices=['CLIPScore', 'CIDEr', 'RefCLIPScore', 'CIDErCLIPScore'])
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--annotation_folder', type=str)
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    parser.add_argument('--force_rl_start', action='store_true')
    
    args = parser.parse_args()
    print(args)

    print('Meshed-Memory Transformer Training')

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # Pipeline for image regions
    image_field = ImageDetectionsFieldWithID(detections_path=args.features_path, max_detections=50, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits

    if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
        text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))

    # Model and dataloaders
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': args.m})
    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    ref_caps_train = list(train_dataset.text)
    if args.reward in ['CIDEr', 'CIDErCLIPScore']:
        cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
    else:
        cider_train = None
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})


    def lambda_lr(s):
        warm_up = args.warmup
        s += 1
        return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)


    # Initial conditions
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    use_rl = False
    best_cider = .0
    best_earlystop_score = .0
    patience = 0
    start_epoch = 0

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = 'saved_models/%s_last.pth' % args.exp_name
        else:
            fname = 'saved_models/%s_best.pth' % args.exp_name

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            try:
                torch.cuda.set_rng_state(data['cuda_rng_state'])
            except:
                print('CHECKPOINT FROM DIFF TORCH VERS, CANT SET RNG STATE')
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            patience = data['patience']
            use_rl = data['use_rl']
            if args.force_rl_start:
                use_rl = True
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                      drop_last=True, shuffle=True)
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers)
        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5,
                                           num_workers=args.workers, shuffle=True)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)

        if not use_rl:
            train_loss = train_xe(model, dataloader_train, optim, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
        else:
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim, cider_train, text_field, args.reward)
            writer.add_scalar('data/train_loss', train_loss, e)
            writer.add_scalar('data/reward', reward, e)
            writer.add_scalar('data/reward_baseline', reward_baseline, e)

        # Validation loss
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field)
        writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, text_field)
        print("Validation scores", scores)
        val_cider = scores['CIDEr']
        val_earlystop = scores[args.use_for_early_stop]
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)
        writer.add_scalar('data/val_clipscore', scores['CLIPScore'], e)
        writer.add_scalar('data/val_refclipscore', scores['RefCLIPScore'], e)

        # Test scores
        scores = evaluate_metrics(model, dict_dataloader_test, text_field)
        print("Test scores", scores)
        writer.add_scalar('data/test_cider', scores['CIDEr'], e)
        writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', scores['ROUGE'], e)
        writer.add_scalar('data/test_clipscore', scores['CLIPScore'], e)
        writer.add_scalar('data/test_refclipscore', scores['RefCLIPScore'], e)

        # Prepare for next epoch
        best = False
        if val_earlystop >= best_earlystop_score:
            best_earlystop_score = val_earlystop
            patience = 0
            best = True
        else:
            patience += 1

        switch_to_rl = False
        exit_train = False
        if patience == 5:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                optim = Adam(model.parameters(), lr=5e-6)
                print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True

        if switch_to_rl and not best:
            data = torch.load('saved_models/%s_best.pth' % args.exp_name)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'])
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))


        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'val_earlystop': val_earlystop,
            'earlystopping_on': args.use_for_early_stop,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'use_rl': use_rl,
        }, 'saved_models/epoch_{}_{}.pth'.format(e, args.exp_name))


        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'val_earlystop': val_earlystop,
            'earlystopping_on': args.use_for_early_stop,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'use_rl': use_rl,
        }, 'saved_models/%s_last.pth' % args.exp_name)

        if best:
            copyfile('saved_models/%s_last.pth' % args.exp_name, 'saved_models/%s_best.pth' % args.exp_name)

        if exit_train:
            writer.close()
            break
