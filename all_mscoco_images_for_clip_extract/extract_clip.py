import argparse
import torch
import clip
import os
import json
import numpy as np
from PIL import Image
import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--captions',
                        default=None,
                        help='captions, one per line.')
    parser.add_argument('--image_dir',
                        default=None,
                        help='image dir')

    parser.add_argument('--clip_model',
                        default='ViT-B/32',
                        choices=['ViT-B/32', 'RN50', 'RN101', 'RN50x4', 'ViT-B/16', 'RN50x16'])

    parser.add_argument('--feature_out_dir',
                        default='clip_features')

    parser.add_argument('--limit',
                        default=-1,
                        type=int)

    parser.add_argument('--load_weights',
                        default=None,
                        type=str)   
    
    args = parser.parse_args()

    if not os.path.exists(args.feature_out_dir):
        os.makedirs(args.feature_out_dir)

    if args.image_dir:
        args.images = [args.image_dir + '/' + x for x in os.listdir(args.image_dir) if '.jpg' in x]
        print('extracting {} image features'.format(len(args.images)))
    else:
        args.images = None

    if args.captions:
        args.captions_out_base = args.feature_out_dir + '/' + args.captions.split('/')[-1].split('.')[0] + '_caption_features~{}'.format(args.clip_model.replace('/',''))
        if args.limit > 0:
            args.captions_out_base += '~limit{}'.format(args.limit)
        if args.load_weights:
            args.captions_out_base += '~initweights'
        print('extracting captions to: {}'.format(args.captions_out_base))
    if args.images:
        args.images_out_base = args.feature_out_dir + '/' + args.image_dir.split('/')[-2] + '_image_features~{}'.format(args.clip_model.replace('/',''))
        if args.limit > 0:
            args.images_out_base += '~limit{}'.format(args.limit)
        if args.load_weights:
            args.images_out_base += '~initweights'
        print('extracting images to: {}'.format(args.images_out_base))

    return args


def extract_all_captions(captions, model, preprocess, args, batch_size=256):
    batches = np.array_split(captions, int(len(captions) / batch_size))
    all_text_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(batches):
            b = clip.tokenize(b).to(args.device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, preprocess, args, batch_size=256):
    batches = np.array_split(images, int(len(images) / batch_size))
    all_image_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(batches):
            b = torch.stack([preprocess(Image.open(im)) for im in b]).to(args.device)
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def main():
    args = parse_args()
    np.random.seed(1)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load(args.clip_model, device=args.device)
    model.eval()
    if args.load_weights:
        print('Loading weights from {}'.format(args.load_weights))
        state = torch.load(args.load_weights)['model_state_dict']
        state.update({'input_resolution': model.input_resolution,
                      'context_length':model.context_length,
                      'vocab_size':model.vocab_size})
        model.load_state_dict(state)

    if args.captions:
        with open(args.captions) as f:
            all_captions = [x.strip() for x in f.readlines()]
        print('extracting text features for {} captions'.format(len(all_captions)))
        unique_captions = list(set(all_captions))
        print('of those, {} were unique'.format(len(unique_captions)))
        if args.limit > 0 :
            np.random.shuffle(unique_captions)
            unique_captions = unique_captions[:args.limit]
            print('limiting to {}'.format(len(unique_captions)))
        all_text_features = extract_all_captions(unique_captions, model, preprocess, args)
        with open(args.captions_out_base + '~cap2row.json', 'w') as f:
            f.write(json.dumps({cap:idx for idx, cap in enumerate(unique_captions)}))
        print('saving {} array'.format(all_text_features.shape))
        np.save(args.captions_out_base + '.npy', all_text_features)

    if args.images:
        ordered_ids, ordered_paths = [], []
        for cpath in args.images:
            cid = '.'.join(cpath.split('/')[-1].split('.')[:-1])
            ordered_ids.append(cid)
            ordered_paths.append(cpath)
        
        if args.limit > 0:
            perm = np.random.permutation(args.limit)
            ordered_ids = [ordered_ids[idx] for idx in perm]
            ordered_paths = [ordered_paths[idx] for idx in perm]
            print('limiting to {}'.format(len(unique_captions)))
            
        all_image_features = extract_all_images(ordered_paths, model, preprocess, args)

        with open(args.images_out_base + '~im2row.json', 'w') as f:
            f.write(json.dumps({cid:idx for idx, cid in enumerate(ordered_ids)}))
        print('saving {} array'.format(all_image_features.shape))
        np.save(args.images_out_base + '.npy', all_image_features)


if __name__ == '__main__':
    main()
