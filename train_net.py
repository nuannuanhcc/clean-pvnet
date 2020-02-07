from lib.config import cfg, args
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network
from lib.evaluators import make_evaluator
import torch.multiprocessing
import neptune


def train(cfg, network):
    if cfg.train.dataset[:4] != 'City':
        torch.multiprocessing.set_sharing_strategy('file_system')
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    if 'Coco' not in cfg.train.dataset:
        evaluator = make_evaluator(cfg)

    begin_epoch = load_model(network, optimizer, scheduler, recorder, cfg.model_dir, resume=cfg.resume)
    # set_lr_scheduler(cfg, scheduler)

    train_loader = make_data_loader(cfg, is_train=True)
    val_loader = make_data_loader(cfg, is_train=False)
    # train_loader = make_data_loader(cfg, is_train=True, max_iter=100)

    global_steps = None
    if cfg.neptune:
        global_steps = {
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }

        neptune.init('hccccccccc/clean-pvnet')
        neptune.create_experiment(cfg.model_dir.split('/')[-1])
        neptune.append_tag('pose')


    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        trainer.train(epoch, train_loader, optimizer, recorder, global_steps)
        scheduler.step()

        if (epoch + 1) % cfg.save_ep == 0:
            save_model(network, optimizer, scheduler, recorder, epoch, cfg.model_dir)

        if (epoch + 1) % cfg.eval_ep == 0:
            if 'Coco' in cfg.train.dataset:
                trainer.val_coco(val_loader, global_steps)
            else:
                trainer.val(epoch, val_loader, evaluator, recorder)

    if cfg.neptune:
        neptune.stop()

    return network


def test(cfg, network):
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    epoch = load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    trainer.val(epoch, val_loader, evaluator)


def main():
    network = make_network(cfg)
    if args.test:
        test(cfg, network)
    else:
        train(cfg, network)


if __name__ == "__main__":
    main()
