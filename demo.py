import argparse
import logging
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from engine import *
from models import *
from voc import *
from losses import MultiLabelSoftMarginLoss, BCEWithLogitsLoss, CombinedLoss

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=224, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[40,80], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--loss-type', default='combined', choices=['softmargin', 'bce', 'combined'],
                    help='loss function type: softmargin | bce | combined')
parser.add_argument('--loss-alpha', default=1.0, type=float, help='weight for base loss')
parser.add_argument('--loss-beta', default=0.0, type=float, help='weight for InfoNCE loss')
parser.add_argument('--loss-gamma', default=1.0, type=float, help='weight for decoder InfoNCE loss')

def main_voc2007():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'logs/voc2007/{current_time}'
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('WILDCAT_Training')
    writer = SummaryWriter(log_dir=log_dir)
    
    use_gpu = torch.cuda.is_available()
    logger.info(f'Using GPU: {use_gpu}')

    logger.info("========== Training Arguments ==========")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg}: {value}")
    logger.info("========================================")
    with open(os.path.join(log_dir, 'config.txt'), 'w') as f:
        for arg, value in sorted(vars(args).items()):
            f.write(f"{arg}: {value}\n")

    train_dataset = Voc2007Classification(args.data, 'trainval', inp_name='data/voc/voc_glove_word2vec.pkl')
    val_dataset = Voc2007Classification(args.data, 'test', inp_name='data/voc/voc_glove_word2vec.pkl')

    num_classes = 20

    model = GCN_DETR_Resnet(
        num_classes=num_classes,
        t=0.4,
        pretrained=True,
        adj_file='data/voc/voc_adj.pkl',
        d_model=256,
        num_heads=4,
    )

    if args.loss_type == 'softmargin':
        criterion = MultiLabelSoftMarginLoss()
    elif args.loss_type == 'bce':
        criterion = BCEWithLogitsLoss()
    elif args.loss_type == 'combined':
        criterion = CombinedLoss(base_loss='softmargin', alpha=args.loss_alpha, beta=args.loss_beta, gamma=args.loss_gamma)
    else:
        raise ValueError('Unknown loss type')

    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {
        'batch_size': args.batch_size, 
        'image_size': args.image_size, 
        'max_epochs': args.epochs,
        'evaluate': args.evaluate, 
        'resume': args.resume, 
        'num_classes': num_classes,
        'log_dir': log_dir,
        'difficult_examples': True,
        'save_model_path': 'checkpoint/voc2007-detrgatn/',
        'workers': args.workers,
        'epoch_step': args.epoch_step,
        'lr': args.lr,
        'logger': logger, 
        'writer': writer  
    }
    
    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)
    
    writer.close()

if __name__ == '__main__':
    main_voc2007()