'''
Predicts without GT needed.
'''
import random
from data import ImageDetectionsField, TextField, RawField
from data.dataset import DictionaryDataset
from data.example import Example
from data import COCO, DataLoader
import evaluation
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import h5py
import pprint
import json

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

def predict_captions(model, dataloader, text_field):
    import itertools
    model.eval()
    image_id_to_pred = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                image_id_to_pred[gts_i[0].replace('placeholder for ', '')] = gen_i
            pbar.update()

    return image_id_to_pred


if __name__ == '__main__':
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--model_ckpt', type=str)
    parser.add_argument('--limit', help='limit to this many images if > 1, debug option', default=-1, type=int)
    
    args = parser.parse_args()

    print('Meshed-Memory Transformer Predictor')

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)
    f = h5py.File(args.features_path, 'r')
    def clean_key(k):
        for clean in ['_features', '_cls_prob', '_boxes']:
            if k[-len(clean):] == clean:
                return k[:-len(clean)]
        return k
    # '91852718_features', '918545497_boxes', '918545497_cls_prob'
    all_image_ids = list(set([clean_key(k) for k in f.keys()]))
    if args.limit > 0:
        all_image_ids = all_image_ids[:args.limit]

    examples = [Example.fromdict({'image': i, 'text': 'placeholder for {}'.format(i)}) for i in all_image_ids]

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))

    # Model and dataloaders
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': 40})
    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    data = torch.load(args.model_ckpt)
    model.load_state_dict(data['state_dict'])
    dict_dataset = DictionaryDataset(examples, {'image': image_field, 'text': RawField()}, key_fields='image')
    dict_dataloader = DataLoader(dict_dataset, batch_size=args.batch_size, num_workers=args.workers)

    image_id2prediction = predict_captions(model, dict_dataloader, text_field)
    with open('image_id_to_prediction~{}~{}.json'.format(args.features_path.split('/')[-1].split('.')[0],
                                                         args.model_ckpt.split('/')[-1].split('.')[0]),
              'w') as f:
        f.write(json.dumps(image_id2prediction))
