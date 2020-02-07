import time
import datetime
import torch
import tqdm
from torch.nn import DataParallel
import numpy as np
from lib.config import cfg, args
from lib.utils.transform_utils import transform_preds
import neptune

class Trainer(object):
    def __init__(self, network):
        network = network.cuda()
        network = DataParallel(network)
        self.network = network

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        for k in batch:
            if k == 'meta':
                continue
            if isinstance(batch[k], tuple):
                batch[k] = [b.cuda() for b in batch[k]]
            else:
                batch[k] = batch[k].cuda()
        return batch

    def train(self, epoch, data_loader, optimizer, recorder, global_steps):
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()
        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1
            recorder.step += 1

            # batch = self.to_cuda(batch)
            output, loss, loss_stats, image_stats = self.network(batch)

            # training stage: loss; optimizer; scheduler
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
            optimizer.step()

            # data recording stage: loss_stats, time, image_stats
            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % 20 == 0 or iteration == (max_iter - 1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record('train')

                if global_steps:
                    neptune_step = global_steps['train_global_steps']
                    neptune.send_metric('train_loss', neptune_step, recorder.loss_stats['loss'].avg)
                    neptune.send_metric('train_loss_seg', neptune_step, recorder.loss_stats['seg_loss'].avg)
                    neptune.send_metric('train_loss_vote', neptune_step, recorder.loss_stats['vote_loss'].avg)
                    global_steps['train_global_steps'] = neptune_step + 1


    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)
        for batch in tqdm.tqdm(data_loader):
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].cuda()

            with torch.no_grad():
                output, loss, loss_stats, image_stats = self.network(batch)
                if evaluator is not None:
                    evaluator.evaluate(output, batch)

            loss_stats = self.reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

        loss_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        print(loss_state)

        if evaluator is not None:
            result = evaluator.summarize()
            val_loss_stats.update(result)

        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)


    def val_coco(self, data_loader, global_steps=None):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        num_samples = len(data_loader)
        all_preds = np.zeros((num_samples, cfg.num_joints, 3),
                             dtype=np.float32)
        all_boxes = np.zeros((num_samples, 6))
        image_path = []
        filenames = []
        imgnums = []
        idx = 0
        with torch.no_grad():
            for batch, meta in tqdm.tqdm(data_loader):
                for k in batch:
                    if k != 'meta':
                        batch[k] = batch[k].cuda()
                output, loss, loss_stats, image_stats = self.network(batch)
                num_images = batch['inp'].size(0)
                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                score = meta['score'].numpy()
                preds = output['kpt_2d'][:, :, 0:2].cpu().numpy().copy()
                for i in range(output['kpt_2d'].shape[0]):
                    preds[i] = transform_preds(output['kpt_2d'][:, :, 0:2][i], c[i], s[i],
                                               [192, 256])
                all_preds[idx:idx + num_images, :, 0:2] = preds
                # all_preds[idx:idx + num_images, :, 2:3] = maxvals
                # double check this all_boxes parts
                all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
                all_boxes[idx:idx + num_images, 5] = score
                image_path.extend(meta['image'])
                idx += num_images

                loss_stats = self.reduce_loss_stats(loss_stats)
                for k, v in loss_stats.items():
                    val_loss_stats.setdefault(k, 0)
                    val_loss_stats[k] += v

        name_values, perf_indicator = data_loader.dataset.evaluate(
                    cfg, all_preds, cfg.result_dir, all_boxes, image_path,
                    filenames, imgnums)

        if global_steps:
            neptune_step = global_steps['valid_global_steps']
            neptune.send_metric('valid_loss', neptune_step, (val_loss_stats['loss']/num_samples).item())
            neptune.send_metric('valid_loss_seg', neptune_step, (val_loss_stats['seg_loss']/num_samples).item())
            neptune.send_metric('valid_loss_vote', neptune_step, (val_loss_stats['vote_loss']/num_samples).item())
            for k, v in name_values.items():
                neptune.send_metric(k, neptune_step, v)
            global_steps['valid_global_steps'] = neptune_step + 1
