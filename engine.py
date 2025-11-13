import os
import shutil
import time
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
from util import *

tqdm.monitor_interval = 0
class Engine(object):
    def __init__(self, state={}):
        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('image_size') is None:
            self.state['image_size'] = 224

        if self._state('batch_size') is None:
            self.state['batch_size'] = 64

        if self._state('workers') is None:
            self.state['workers'] = 25

        if self._state('device_ids') is None:
            self.state['device_ids'] = None

        if self._state('evaluate') is None:
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 90

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []

        # meters
        self.state['meter_loss'] = tnt.meter.AverageValueMeter()
        # time measure
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()
        # display parameters
        if self._state('use_pb') is None:
            self.state['use_pb'] = True
        if self._state('print_freq') is None:
            self.state['print_freq'] = 0

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        map = 100 * self.state['ap_meter'].value().mean()
        loss = self.state['meter_loss'].value()[0]
        OP, OR, OF1, CP, CR, CF1 = self.state['ap_meter'].overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.state['ap_meter'].overall_topk(3)
        
        epoch = self.state.get('epoch', 0)
        phase = 'train' if training else 'val'

        logger = self._state('logger')
        writer = self._state('writer')

        if writer is not None:
            writer.add_scalar(f'{phase}/loss', loss, epoch)
            writer.add_scalar(f'{phase}/mAP', map, epoch)
            writer.add_scalar(f'{phase}/OP', OP, epoch)
            writer.add_scalar(f'{phase}/OR', OR, epoch)
            writer.add_scalar(f'{phase}/OF1', OF1, epoch)
            writer.add_scalar(f'{phase}/CP', CP, epoch)
            writer.add_scalar(f'{phase}/CR', CR, epoch)
            writer.add_scalar(f'{phase}/CF1', CF1, epoch)
            writer.add_scalar(f'{phase}/OP_3', OP_k, epoch)
            writer.add_scalar(f'{phase}/OR_3', OR_k, epoch)
            writer.add_scalar(f'{phase}/OF1_3', OF1_k, epoch)
            writer.add_scalar(f'{phase}/CP_3', CP_k, epoch)
            writer.add_scalar(f'{phase}/CR_3', CR_k, epoch)
            writer.add_scalar(f'{phase}/CF1_3', CF1_k, epoch)
        if logger is not None:
            logger.info(f'[{phase}] Epoch: {epoch} Loss: {loss:.4f} mAP: {map:.3f} '
                        f'OP: {OP:.4f} OR: {OR:.4f} OF1: {OF1:.4f} '
                        f'CP: {CP:.4f} CR: {CR:.4f} CF1: {CF1:.4f} '
                        f'OP_3: {OP_k:.4f} OR_3: {OR_k:.4f} OF1_3: {OF1_k:.4f} '
                        f'CP_3: {CP_k:.4f} CR_3: {CR_k:.4f} CF1_3: {CF1_k:.4f}')

        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}\t'
                      'mAP {map:.3f}'.format(self.state['epoch'], loss=loss, map=map))
                print('OP: {OP:.4f}\t'
                      'OR: {OR:.4f}\t'
                      'OF1: {OF1:.4f}\t'
                      'CP: {CP:.4f}\t'
                      'CR: {CR:.4f}\t'
                      'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
            else:
                print('Test: \t Loss {loss:.4f}\t mAP {map:.3f}'.format(loss=loss, map=map))
                print('OP: {OP:.4f}\t'
                      'OR: {OR:.4f}\t'
                      'OF1: {OF1:.4f}\t'
                      'CP: {CP:.4f}\t'
                      'CR: {CR:.4f}\t'
                      'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
                print('OP_3: {OP:.4f}\t'
                      'OR_3: {OR:.4f}\t'
                      'OF1_3: {OF1:.4f}\t'
                      'CP_3: {CP:.4f}\t'
                      'CR_3: {CR:.4f}\t'
                      'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))

        return map

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
    
        try:
            Engine.on_end_batch(self, training, model, criterion, data_loader, optimizer, display=False)
        except Exception:
            pass

        try:
            out_for_ap = self.state.get('output_valid', None)
            tgt_for_ap = self.state.get('target_gt_valid', None)
            if out_for_ap is None or tgt_for_ap is None or out_for_ap.numel() == 0:
                out_for_ap = self.state.get('output', None)
                tgt_for_ap = self.state.get('target_gt', None)

            if out_for_ap is not None and tgt_for_ap is not None:
     
                out_cpu = out_for_ap.detach().cpu()
                tgt_cpu = tgt_for_ap.detach().cpu()
                if out_cpu.dim() > 2:
                    out_cpu = out_cpu.contiguous().view(-1, out_cpu.size(-1))
                if tgt_cpu.dim() > 2:
                    tgt_cpu = tgt_cpu.contiguous().view(-1, tgt_cpu.size(-1))
              
                if out_cpu.dim() == 2 and tgt_cpu.dim() == 2 and out_cpu.size() == tgt_cpu.size():
                    self.state['ap_meter'].add(out_cpu, tgt_cpu)
        except Exception:
      
            pass

        base_loss = self.state.get('base_loss', None)
        nce_loss = self.state.get('nce_loss', None)
        decoder_nce = self.state.get('decoder_nce', None)
        if base_loss is not None:
            if isinstance(base_loss, torch.Tensor):
                base_loss = base_loss.detach().cpu().item()
            self.state['meter_base_loss'].add(base_loss)
        if nce_loss is not None:
            if isinstance(nce_loss, torch.Tensor):
                nce_loss = nce_loss.detach().cpu().item()
            self.state['meter_nce_loss'].add(nce_loss)
        if decoder_nce is not None:
            if isinstance(decoder_nce, torch.Tensor):
                decoder_nce = decoder_nce.detach().cpu().item()
            self.state['meter_decoder_nce'].add(decoder_nce)

        if display and self.state.get('print_freq', 0) != 0 and (self.state.get('iteration', 0) % self.state.get('print_freq', 1) == 0):
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))    
    
    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):

        imgs = self.state.get('input_imgs', None)    
        labels = self.state.get('target', None)    
        target_gt = self.state.get('target_gt', None)
        status = self.state.get('status', None)     
        device = next(model.parameters()).device

        if imgs is None or target_gt is None or status is None:
            return
        imgs = imgs.to(device)
        labels = labels.to(device)
        target_gt = target_gt.to(device)
        status = status.to(device)

        is_combined = hasattr(criterion, 'forward') and 'return_parts' in criterion.forward.__code__.co_varnames

        if not training:
            model.eval()
            with torch.no_grad():
                out, hs, query_embed = model(imgs)  
        else:
            model.train()
            out, hs, query_embed = model(imgs)  

        B, T, C = out.shape

        out_flat = out.contiguous().view(B * T, C)
        target_flat = target_gt.contiguous().view(B * T, C)
        status_flat = status.contiguous().view(B * T)

        valid_mask = (status_flat == 0)
        valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(1)

        hs_flat = None
        if hs is not None:
            if isinstance(hs, torch.Tensor):
                if hs.dim() == 4:
                    hs_flat = hs.contiguous().view(B * T, C, hs.size(-1))
                elif hs.dim() == 3:
                    hs_flat = hs.contiguous().view(B * T, C, -1)
            else:
                hs_flat = None

        if valid_idx.numel() > 0:
            out_valid = out_flat[valid_idx]
            target_valid = target_flat[valid_idx]
            hs_valid = hs_flat[valid_idx] if hs_flat is not None else None
        else:
            out_valid = torch.zeros((0, C), device=device)
            target_valid = torch.zeros((0, C), device=device)
            hs_valid = None

        if not training:
            if is_combined:
                loss, base_loss, nce_loss, decoder_nce = criterion(out_valid, target_valid, hs=hs_valid, query_embed=query_embed, return_parts=True)
                self.state['loss'] = loss
                self.state['base_loss'] = base_loss
                self.state['nce_loss'] = nce_loss
                self.state['decoder_nce'] = decoder_nce
            else:
                self.state['loss'] = criterion(out_valid, target_valid)
        else:
            if is_combined:
                loss, base_loss, nce_loss, decoder_nce = criterion(out_valid, target_valid, hs=hs_valid, query_embed=query_embed, return_parts=True)
                self.state['loss'] = loss
                self.state['base_loss'] = base_loss
                self.state['nce_loss'] = nce_loss
                self.state['decoder_nce'] = decoder_nce
            else:
                self.state['loss'] = criterion(out_valid, target_valid)

            optimizer.zero_grad()
            self.state['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        self.state['output'] = out_flat.detach()       
        self.state['target_gt'] = target_flat.detach() 

        self.state['output_valid'] = out_valid.detach()       
        self.state['target_gt_valid'] = target_valid.detach()  

        try:
            if self.state['output_valid'].numel() > 0:
                self.state['ap_meter'].add(self.state['output_valid'].cpu(), self.state['target_gt_valid'].cpu())
            else:
                out_f = self.state['output'].cpu()
                tgt_f = self.state['target_gt'].cpu()
                if out_f.dim() == 2 and tgt_f.dim() == 2 and out_f.size() == tgt_f.size():
                    self.state['ap_meter'].add(out_f, tgt_f)
        except Exception:
            pass

        self.state['hs'] = hs_valid
        self.state['query_embed'] = query_embed
        
    def init_learning(self, model, criterion):

        if self._state('train_transform') is None:
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            self.state['train_transform'] = transforms.Compose([
                MultiScaleCrop(self.state['image_size'], scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        if self._state('val_transform') is None:
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            self.state['val_transform'] = transforms.Compose([
                Warp(self.state['image_size']),
                transforms.ToTensor(),
                normalize,
            ])

        self.state['best_score'] = 0

    def learning(self, model, criterion, train_dataset, val_dataset, optimizer=None):

        self.init_learning(model, criterion)

        # define train and val transform
        train_dataset.transform = self.state['train_transform']
        train_dataset.target_transform = self._state('train_target_transform')
        val_dataset.transform = self.state['val_transform']
        val_dataset.target_transform = self._state('val_target_transform')

        # data loading code
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.state['batch_size'], shuffle=True,
                                                   num_workers=self.state['workers'])

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.state['batch_size'], shuffle=False,
                                                 num_workers=self.state['workers'])

        # optionally resume from a checkpoint
        if self._state('resume') is not None:
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['resume'])
                self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score'] = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.state['evaluate'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))


        if self.state['use_gpu']:
            train_loader.pin_memory = True
            val_loader.pin_memory = True
            cudnn.benchmark = True


            model = torch.nn.DataParallel(model, device_ids=self.state['device_ids']).cuda()


            criterion = criterion.cuda()

        if self.state['evaluate']:
            self.validate(val_loader, model, criterion)
            return

        # TODO define optimizer

        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            lr = self.adjust_learning_rate(optimizer)
            print('lr:',lr)

            # train for one epoch
            self.train(train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            prec1 = self.validate(val_loader, model, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.state['best_score']
            self.state['best_score'] = max(prec1, self.state['best_score'])
            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self._state('arch'),
                'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(),
                'best_score': self.state['best_score'],
            }, is_best)

            print(' *** best={best:.3f}'.format(best=self.state['best_score']))
        return self.state['best_score']

    def train(self, data_loader, model, criterion, optimizer, epoch):

        # switch to train mode
        model.train()

        self.on_start_epoch(True, model, criterion, data_loader, optimizer)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Training')

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(True, model, criterion, data_loader, optimizer)

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(non_blocking=True)

            self.on_forward(True, model, criterion, data_loader, optimizer)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(True, model, criterion, data_loader, optimizer)

        self.on_end_epoch(True, model, criterion, data_loader, optimizer)

    def validate(self, data_loader, model, criterion):

        # switch to evaluate mode
        model.eval()

        self.on_start_epoch(False, model, criterion, data_loader)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Test')

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(False, model, criterion, data_loader)

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(non_blocking=True)

            self.on_forward(False, model, criterion, data_loader)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(False, model, criterion, data_loader)

        score = self.on_end_epoch(False, model, criterion, data_loader)

        return score

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if self._state('save_model_path') is not None:
            filename_ = filename
            filename = os.path.join(self.state['save_model_path'], filename_)
            if not os.path.exists(self.state['save_model_path']):
                os.makedirs(self.state['save_model_path'])
        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)
        if is_best:
            filename_best = 'model_best.pth.tar'
            if self._state('save_model_path') is not None:
                filename_best = os.path.join(self.state['save_model_path'], filename_best)
            shutil.copyfile(filename, filename_best)
            if self._state('save_model_path') is not None:
                if self._state('filename_previous_best') is not None:
                    os.remove(self._state('filename_previous_best'))
                filename_best = os.path.join(self.state['save_model_path'], 'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']))
                shutil.copyfile(filename, filename_best)
                self.state['filename_previous_best'] = filename_best

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr_list = []
        decay = 0.1 if sum(self.state['epoch'] == np.array(self.state['epoch_step'])) > 0 else 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay
            lr_list.append(param_group['lr'])
        return np.unique(lr_list)


class MultiLabelMAPEngine(Engine):
    def __init__(self, state):
        Engine.__init__(self, state)
        if self._state('difficult_examples') is None:
            self.state['difficult_examples'] = False
        self.state['ap_meter'] = AveragePrecisionMeter(self.state['difficult_examples'])

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        Engine.on_start_epoch(self, training, model, criterion, data_loader, optimizer)
        self.state['ap_meter'].reset()
        if 'meter_base_loss' not in self.state:
            self.state['meter_base_loss'] = tnt.meter.AverageValueMeter()
        if 'meter_nce_loss' not in self.state:
            self.state['meter_nce_loss'] = tnt.meter.AverageValueMeter()
        if 'meter_decoder_nce' not in self.state:
            self.state['meter_decoder_nce'] = tnt.meter.AverageValueMeter()
        self.state['meter_base_loss'].reset()
        self.state['meter_nce_loss'].reset()
        self.state['meter_decoder_nce'].reset()
        
    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        ap_per_class = self.state['ap_meter'].value()
        map = 100 * self.state['ap_meter'].value().mean()
        loss = self.state['meter_loss'].value()[0]
        OP, OR, OF1, CP, CR, CF1 = self.state['ap_meter'].overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.state['ap_meter'].overall_topk(3)
        
        epoch = self.state.get('epoch', 0)
        phase = 'train' if training else 'val'

        logger = self._state('logger')
        writer = self._state('writer')

        base_loss = self.state['meter_base_loss'].value()[0] if 'meter_base_loss' in self.state else None
        nce_loss = self.state['meter_nce_loss'].value()[0] if 'meter_nce_loss' in self.state else None
        decoder_nce = self.state['meter_decoder_nce'].value()[0] if 'meter_decoder_nce' in self.state else None
        
        if writer is not None:
            writer.add_scalar(f'{phase}/loss', loss, epoch)
            writer.add_scalar(f'{phase}/mAP', map, epoch)
            writer.add_scalar(f'{phase}/OP', OP, epoch)
            writer.add_scalar(f'{phase}/OR', OR, epoch)
            writer.add_scalar(f'{phase}/OF1', OF1, epoch)
            writer.add_scalar(f'{phase}/CP', CP, epoch)
            writer.add_scalar(f'{phase}/CR', CR, epoch)
            writer.add_scalar(f'{phase}/CF1', CF1, epoch)
            writer.add_scalar(f'{phase}/OP_3', OP_k, epoch)
            writer.add_scalar(f'{phase}/OR_3', OR_k, epoch)
            writer.add_scalar(f'{phase}/OF1_3', OF1_k, epoch)
            writer.add_scalar(f'{phase}/CP_3', CP_k, epoch)
            writer.add_scalar(f'{phase}/CR_3', CR_k, epoch)
            writer.add_scalar(f'{phase}/CF1_3', CF1_k, epoch)
            if base_loss is not None:
                writer.add_scalar(f'{phase}/base_loss', base_loss, epoch)
            if nce_loss is not None:
                writer.add_scalar(f'{phase}/nce_loss', nce_loss, epoch)
            if decoder_nce is not None:
                writer.add_scalar(f'{phase}/decoder_nce', decoder_nce, epoch)
        if logger is not None:
            logger.info(f'[{phase}] Epoch: {epoch} Loss: {loss:.4f} mAP: {map:.3f} '
                        f'OP: {OP:.4f} OR: {OR:.4f} OF1: {OF1:.4f} '
                        f'CP: {CP:.4f} CR: {CR:.4f} CF1: {CF1:.4f} '
                        f'OP_3: {OP_k:.4f} OR_3: {OR_k:.4f} OF1_3: {OF1_k:.4f} '
                        f'CP_3: {CP_k:.4f} CR_3: {CR_k:.4f} CF1_3: {CF1_k:.4f}' +
                        (f' BaseLoss: {base_loss:.4f}' if base_loss is not None else '') +
                        (f' InfoNCE: {nce_loss:.4f}' if nce_loss is not None else '') +
                        (f' DecoderNCE: {decoder_nce:.4f}' if decoder_nce is not None else ''))

        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}\t'
                      'mAP {map:.3f}'.format(self.state['epoch'], loss=loss, map=map))
                print('OP: {OP:.4f}\t'
                      'OR: {OR:.4f}\t'
                      'OF1: {OF1:.4f}\t'
                      'CP: {CP:.4f}\t'
                      'CR: {CR:.4f}\t'
                      'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
            else:
                print('Test: \t Loss {loss:.4f}\t mAP {map:.3f}'.format(loss=loss, map=map))
                print('OP: {OP:.4f}\t'
                      'OR: {OR:.4f}\t'
                      'OF1: {OF1:.4f}\t'
                      'CP: {CP:.4f}\t'
                      'CR: {CR:.4f}\t'
                      'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
                print('OP_3: {OP:.4f}\t'
                      'OR_3: {OR:.4f}\t'
                      'OF1_3: {OF1:.4f}\t'
                      'CP_3: {CP:.4f}\t'
                      'CR_3: {CR:.4f}\t'
                      'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))
                
                ap_per_class = ap_per_class.cpu().numpy()
                print("Per-class AP:")
                for i, ap_val in enumerate(ap_per_class):
                    print(f"Class {i}: {ap_val:.4f}")
                    
        return map

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        self.state['target_gt'] = self.state['target'].clone()
        self.state['target'][self.state['target'] == 0] = 1
        self.state['target'][self.state['target'] == -1] = 0

        input = self.state['input']
        self.state['input'] = input[0]
        self.state['name'] = input[1]

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        try:
            Engine.on_end_batch(self, training, model, criterion, data_loader, optimizer, display=False)
        except Exception:
            pass

        try:
            out_for_ap = self.state.get('output_valid', None)
            tgt_for_ap = self.state.get('target_gt_valid', None)

            if out_for_ap is None or tgt_for_ap is None or (isinstance(out_for_ap, torch.Tensor) and out_for_ap.numel() == 0):
                out_for_ap = self.state.get('output', None)
                tgt_for_ap = self.state.get('target_gt', None)

            if out_for_ap is not None and tgt_for_ap is not None:
                out_cpu = out_for_ap.detach().cpu() if isinstance(out_for_ap, torch.Tensor) else torch.tensor(out_for_ap)
                tgt_cpu = tgt_for_ap.detach().cpu() if isinstance(tgt_for_ap, torch.Tensor) else torch.tensor(tgt_for_ap)

                if out_cpu.dim() > 2:
                    out_cpu = out_cpu.contiguous().view(-1, out_cpu.size(-1))
                if tgt_cpu.dim() > 2:
                    tgt_cpu = tgt_cpu.contiguous().view(-1, tgt_cpu.size(-1))

                if out_cpu.dim() == 2 and tgt_cpu.dim() == 2 and out_cpu.size() == tgt_cpu.size():
                    self.state['ap_meter'].add(out_cpu, tgt_cpu)
        except Exception:
            pass

        base_loss = self.state.get('base_loss', None)
        nce_loss = self.state.get('nce_loss', None)
        decoder_nce = self.state.get('decoder_nce', None)
        if base_loss is not None:
            if isinstance(base_loss, torch.Tensor):
                base_loss = base_loss.detach().cpu().item()
            self.state['meter_base_loss'].add(base_loss)
        if nce_loss is not None:
            if isinstance(nce_loss, torch.Tensor):
                nce_loss = nce_loss.detach().cpu().item()
            self.state['meter_nce_loss'].add(nce_loss)
        if decoder_nce is not None:
            if isinstance(decoder_nce, torch.Tensor):
                decoder_nce = decoder_nce


class GCNMultiLabelMAPEngine(MultiLabelMAPEngine):
    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        feature_tensor = self.state['feature'].float()
        target_tensor = self.state['target'].float()
        input_tensor = self.state['input'].float().detach()

        device = next(model.parameters()).device
        feature_tensor = feature_tensor.to(device)
        target_tensor = target_tensor.to(device)
        input_tensor = input_tensor.to(device)

        output, hs, query_embed = model(feature_tensor, input_tensor)
        self.state['output'] = output
        self.state['hs'] = hs
        self.state['query_embed'] = query_embed

        is_combined = hasattr(criterion, 'forward') and 'return_parts' in criterion.forward.__code__.co_varnames

        if not training:
            model.eval()
            with torch.no_grad():
                if is_combined:
                    loss, base_loss, nce_loss, decoder_nce = criterion(
                        output, target_tensor, hs=hs, query_embed=query_embed, return_parts=True)
                    self.state['loss'] = loss
                    self.state['base_loss'] = base_loss
                    self.state['nce_loss'] = nce_loss
                    self.state['decoder_nce'] = decoder_nce
                else:
                    self.state['loss'] = criterion(output, target_tensor)
        else:
            model.train()
            if is_combined:
                loss, base_loss, nce_loss, decoder_nce = criterion(
                    output, target_tensor, hs=hs, query_embed=query_embed, return_parts=True)
                self.state['loss'] = loss
                self.state['base_loss'] = base_loss
                self.state['nce_loss'] = nce_loss
                self.state['decoder_nce'] = decoder_nce
            else:
                self.state['loss'] = criterion(output, target_tensor)

            optimizer.zero_grad()
            self.state['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            if hasattr(model, 'module'):
                qe = model.module.query_embed
            else:
                qe = model.query_embed

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        self.state['target_gt'] = self.state['target'].clone()
        self.state['target'][self.state['target'] == 0] = 1
        self.state['target'][self.state['target'] == -1] = 0

        input = self.state['input']
        self.state['feature'] = input[0]
        self.state['out'] = input[1]
        self.state['input'] = input[2]