'''
Predicts without GT needed.
'''
import random
from data import ImageDetectionsField, TextField, RawField
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

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def predict_captions(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)

            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i.strip(), ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    return gen, gts


if __name__ == '__main__':
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--model_ckpt', type=str)
    
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
    dict_dataset = DictionaryDataset(examples, {'image': image_field, 'text': RawField()})
    dict_dataloader = DataLoader(dict_dataset, batch_size=args.batch_size, num_workers=args.workers)

    preds, refs = predict_captions(model, dict_dataloader_test, text_field)

    with open('preds.json', 'w') as f:
        f.write(preds)
    with open('refs.json', 'w') as f:
        f.write(refs)
