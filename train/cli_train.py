
import argparse, yaml, torch, os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..data.dataset import AudioDataset
from ..data.features import extract_features, apply_augmentation
from ..data.sampler import create_samplers
from ..models.transformer_ser import ImprovedTransformerSER
from ..train.trainer import Trainer
from ..utils.seed import set_global_seed

def load_paths_labels(root):
    mapping = {'IEO':'Neutral','DFA':'Disgust','IOM':'Happy','IWL':'Angry',
               'TAI':'Fear','TIE':'Sad','ITS':'Neutral','ITH':'Happy',
               'MTI':'Angry','WSI':'Disgust','TSI':'Fear','MMW':'Sad'}
    paths, labels = [], []
    for f in os.listdir(root):
        if f.endswith('.wav'):
            emo = mapping.get(f.split('_')[1], None)
            if emo:
                paths.append(os.path.join(root,f)); labels.append(emo)
    return paths, labels

def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    set_global_seed(cfg['train']['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    paths, labels = load_paths_labels(cfg['data']['root'])
    le = LabelEncoder(); y = le.fit_transform(labels)
    p_train,p_test,y_train,y_test = train_test_split(paths,y,test_size=0.2,stratify=y,random_state=42)
    p_train,p_val,y_train,y_val = train_test_split(p_train,y_train,test_size=0.15,stratify=y_train,random_state=42)

    train_ds = AudioDataset(p_train,y_train,cfg['data']['max_len'],True,extract_features,apply_augmentation)
    val_ds   = AudioDataset(p_val,y_val,cfg['data']['max_len'],False,extract_features)
    train_loader = torch.utils.data.DataLoader(train_ds,batch_size=cfg['data']['batch_size'],
                                               sampler=create_samplers(y_train,y_val)[0],
                                               num_workers=cfg['train']['num_workers'])
    val_loader = torch.utils.data.DataLoader(val_ds,batch_size=cfg['data']['batch_size'],
                                             sampler=create_samplers(y_train,y_val)[1],
                                             num_workers=cfg['train']['num_workers'])
    model = ImprovedTransformerSER(128+7+12, cfg['model']['nhead'], cfg['model']['dim_feedforward'],
                                   cfg['model']['num_layers'], len(le.classes_), cfg['data']['max_len'],
                                   cfg['model']['dropout']).to(device)
    crit = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['optim']['lr'], weight_decay=cfg['optim']['weight_decay'])
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    sch = CosineAnnealingWarmRestarts(opt, T_0=cfg['optim']['scheduler']['T_0'],
                                      T_mult=cfg['optim']['scheduler']['T_mult'],
                                      eta_min=cfg['optim']['scheduler']['eta_min'])
    Trainer(model,crit,opt,sch,device).fit(train_loader,val_loader,cfg['train']['epochs'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args = parser.parse_args()
    main(args.config)
