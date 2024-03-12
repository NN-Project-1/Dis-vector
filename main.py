import os
import yaml
import argparse
from torch.backends import cudnn

from Trainer import trainer
from data_loader import get_loader
from data_preprocessing import preprocess_data
from utils import Dict2Class


def main(config, args):
    cudnn.benchmark = True

    if args.stage == 0:
        preprocess_data(config)
    elif args.stage == 1:
        data_loader = get_loader(config)

        for batch_idx, batch in enumerate(data_loader):
            spk_id_org, spmel_gt, content_input, pitch_input,  len_crop = batch
        trainer = trainer(data_loader, args, config)
        trainerr.train()
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iters', type=int, default=50000)
    parser.add_argument('--resume_iters', type=int, default=0)
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--ckpt_save_step', type=int, default=1000)
    parser.add_argument('--stage', type=int, default=1, help='0: preprocessing; 1: training')
    parser.add_argument('--config_name', type=str, default='spsp2-large')
    parser.add_argument('--model_type', type=str, default='G', help='G: generator) 

    config = yaml.safe_load(open(os.path.join('configs', f'{args.config_name}.yaml'), 'r'))
    config = Dict2Class(config)
    

    main(config, args)
