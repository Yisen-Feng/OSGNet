import argparse
import os
import time
import torch
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed, ReferringRecall, valid_one_epoch_nlq_singlegpu
from libs.datasets.data_utils import trivial_batch_collator


################################################################################
def main(args):
    """0. load config"""
    # sanity check
    os.environ["LOCAL_RANK"]="0"
    os.environ["CUDA_VISIBLE_DEVICES"] =  str(args.gpu)
    print(args.config)
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['val_split']) > 0, "Test set must be specified!"

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        collate_fn=trivial_batch_collator,
        batch_size=cfg['loader']['batch_size'],
        num_workers=cfg['loader']['num_workers'],
        shuffle=False,
    )

    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])

    """4. load ckpt"""
    paths=[]
    if os.path.isdir(args.resume):
        for file in os.listdir(args.resume):
            # if file in ['epoch_001.pth.tar','epoch_000.pth.tar','epoch_002.pth.tar','epoch_004.pth.tar','epoch_003.pth.tar','epoch_005.pth.tar','epoch_006.pth.tar']:
            #     continue
            if file.split('.')[-1]=='tar':
                paths.append(os.path.join(args.resume,file))
    else:
        paths.append(args.resume)
    for path in paths:
    # load ckpt, reset epoch / best rmse
        checkpoint = torch.load(path, map_location="cpu")
        # args.start_epoch = checkpoint['epoch'] + 1
        for key in checkpoint['state_dict'].keys():
            if key.startswith('module'):
                loaded_model = torch.nn.DataParallel(model)
                loaded_model.load_state_dict(checkpoint['state_dict'])

                # 去除 "module." 前缀，得到单 GPU 模型
                model = loaded_model.module
            else:
                model.load_state_dict(checkpoint['state_dict'])
                # model.load_state_dict(checkpoint['state_dict_ema'])
                # print('success load ema')
                # model.load_state_dict(checkpoint['state_dict'],strict=False)
            break
        # also load the optimizer / scheduler if necessary
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{:s}' (epoch {:d})".format(
            path, checkpoint['epoch']
        ))
        model.to(torch.device("cuda:0"))

        # set up evaluator
        det_eval = ReferringRecall(dataset=cfg["track"],gt_file=cfg["dataset"]["json_file"])

        output_file = None
        if args.save:
            output_file = [
                os.path.join(os.path.split(args.resume)[0], 'nlq_predictions_epoch_val_top10_%d.json'%checkpoint['epoch']),
                os.path.join(os.path.split(args.resume)[0], 'nlq_predictions_epoch_val_top10_%d_noscore.json'%checkpoint['epoch'])
            ]
            if 'use_Bayesian' in cfg['dataset']:
                print('use_Bayesian')
                output_file = [
                os.path.join(os.path.split(args.resume)[0], 'nlq_predictions_epoch_val_top10_%d_Bayesian.json'%checkpoint['epoch']),
                os.path.join(os.path.split(args.resume)[0], 'nlq_predictions_epoch_val_top10_%d_noscore_Bayesian.json'%checkpoint['epoch'])
            ]
        """5. Test the model"""
        print("\nStart testing model {:s} ...".format(cfg['model_name']))
        start = time.time()
        max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
        )
        model.max_epoch=max_epochs
        results = valid_one_epoch_nlq_singlegpu(
            val_loader,
            model,
            checkpoint['epoch'],
            evaluator=det_eval,
            output_file=output_file,
            tb_writer=None,
            # print_freq=args.print_freq
        )
        end = time.time()
        print("All done! Total time: {:0.2f} sec".format(end - start))
    return


################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('resume', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--save', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-gpu', '--gpu', default=0, type=int,
                        help='gpu_id')
    args = parser.parse_args()
    main(args)
