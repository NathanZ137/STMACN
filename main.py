from trainer.STMACN_trainer import STMACNTrainer
from argparse import ArgumentParser

parser = ArgumentParser(description='Run')
parser.add_argument('--cfg_file', type=str, default='./config/PeMS08/config.json', help='Config File')
parser.add_argument('--model', type=str, default='STMACN', help='Model Name')
parser.add_argument('--run_type', type=str, default='train', help='train, eval')
args = parser.parse_args()
if __name__ == "__main__":
    if args.model == 'STMACN':
        trainer = STMACNTrainer(cfg_file=args.cfg_file)
    
    if args.run_type == 'train':
        trainer.train()
    elif args.run_type == 'eval':
        trainer.eval()