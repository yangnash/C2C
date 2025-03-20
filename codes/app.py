import random
import os
import numpy as np
from models.vm_models.c2c import C2C
from dataset.com_video_dataset import CompositionVideoDataset
from opts import parser
import yaml
import torch.multiprocessing
from train_models import evaluate
import test as test

# torch.multiprocessing.set_sharing_strategy('file_system')

def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_args(filename, args):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for key, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)

config = parser.parse_args()
config_path = '/opt/data/private/bishe/C2C/codes/config/c2c_vm/c2c_vanilla_tsm.yml'
load_args(config_path, config)
set_seed(config.seed)


dataset_path = config.dataset_path
dataset=CompositionVideoDataset(dataset_path,
                                phase='train',
                                split='compositional-split-natural',
                                tdn_input='tdn' in config.arch,
                                frames_duration=config.num_frames)
model = C2C(dataset,cfg=config)
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(
    torch.load(os.path.join('/opt/data/private/bishe/weight/best.pt')), 
    strict=False
)
model.eval()

attr2idx = dataset.attr2idx
obj2idx = dataset.obj2idx
    # print(text_rep.shape)
pairs_dataset = dataset.pairs
pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                        for attr, obj in pairs_dataset]).cuda()



# all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
#         model, dataset, config)

# preds=torch.max(all_logits, dim=1)[1]
# correct = (preds == all_pair_gt).sum().item()
# total = len(all_pair_gt)
# accuracy = correct / total * 100
# print(f'总体准确率: {accuracy:.2f}%')
labels = {
        i: pair_name
        for i, pair_name in enumerate(dataset.pairs)
}
def predict(video):
    logits=model(video.cuda(), pairs)
    pred = torch.max(logits, dim=1)[1]
    pred = pred.cpu()
    return labels[pred.item()]