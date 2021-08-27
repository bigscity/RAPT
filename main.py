import os
import torch
import random
import logging
import argparse

import numpy as np
import models.model as Model

from loader import load_data
from train import pretrain, train, test


def main():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model', default='Predict', type=str, help='pretrain or predict')
    parser.add_argument('--input_dim', default=129, type=int, help='input dimension')
    parser.add_argument('--hidden_dim', default=256, type=int, help='hidden dimension')
    parser.add_argument('--layer_num', default=1, type=int, help='the number of layers')
    parser.add_argument('--head_num', default=6, type=int, help='the number of head for transformer based backbone')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout for classified layer')
    parser.add_argument('--max_len', default=128, type=int, help='max length of data')
    parser.add_argument('--aggregator', default='GeneralAttention', type=str, help='which aggregator to use')

    # train
    parser.add_argument('--model_path', default=None, type=str, help='model for fine-tune or test')
    parser.add_argument('--epoch_num', default=64, type=int, help='the number of epoch')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--do_train', action='store_true', help='whether to train the model')
    parser.add_argument('--do_test', action='store_true', help='whether to test the model')

    # optimizer
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='l2 regularization')
    parser.add_argument('--step_size', default=1, type=int, help='step size to reduce')
    parser.add_argument('--gamma', default=0.99, type=float, help='reduce ratio')

    # pre train
    parser.add_argument('--margin', default=3, type=float, help='margin for similar prediction')
    parser.add_argument('--mask_p', default=0.3, type=float, help='mask ratio for masked prediction')
    parser.add_argument('--pre_n', default=4, type=int, help='prediction number for masked prediction')
    parser.add_argument('--mode', default='stable', type=str, help='parameter change mode')
    parser.add_argument('--task', default='sp,mp,rc', type=str, help='pre-training task')
    parser.add_argument('--proportion', default='0.2,0.3,0.5', type=str, help='parameter for balance pre-training task')

    # validate
    parser.add_argument('--scores', default='acc,f1,auc', type=str, help='scores to validate the model')
    parser.add_argument('--main_score', default='auc', type=str, help='score to select best model')

    # other
    parser.add_argument('--output_path', default=None, type=str, help='path to save log and model')
    parser.add_argument('--device', default='cuda:0', type=str, help='training device')
    parser.add_argument('--random_seed', default=-1, type=int, help='random seed for reproduction')
    parser.add_argument('--no_stream', action='store_true', help='whether to output to stream')

    args = parser.parse_args()

    # set output path
    if args.model_path is not None and not args.do_train:
        args.output_path = os.path.join('runs', args.model_path)
    else:
        args.output_path = os.path.join('runs', 'RAPT-%s' % args.model)

        base, index = args.output_path, 1
        while os.path.exists(base):
            base = args.output_path + '-' + str(index)
            index = index + 1
        args.output_path = base
        os.makedirs(args.output_path)

    # set logger
    log = logging.getLogger()
    log_format = '[%(asctime)s] [%(levelname)s] [%(filename)s] [Line %(lineno)d] %(message)s'
    log_format = logging.Formatter(log_format, '%Y-%m-%d %H:%M:%S')
    log.setLevel(logging.INFO)

    if args.do_train:
        file_path = os.path.join(args.output_path, 'train.log')
        file_handler = logging.FileHandler(file_path, mode='w')
        file_handler.setFormatter(log_format)
        log.addHandler(file_handler)

    if not args.no_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_format)
        log.addHandler(stream_handler)

    # set random seed for reproduction
    if args.random_seed > 0:
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_seed)
        random.seed(args.random_seed)

    # print parameters for reproduction
    parameters = list()
    for k, v in args.__dict__.items():
        if type(v) == bool and v:
            parameters.append('--{}'.format(k))
        elif type(v) != bool:
            parameters.append('--{} {}'.format(k, v))
    logging.info('Arguments:\n\tpython main.py {}'.format(('\n\t' + ' ' * 15).join(parameters)))
    logging.info('Output path: {}'.format(args.output_path))

    if not torch.cuda.is_available():
        args.device = 'cpu'
        logging.info('CUDA is not available. Changing device to cpu!')

    model = getattr(Model, args.model)(args).to(args.device)

    if args.model_path is not None:
        checkpoint = torch.load(os.path.join('runs', args.model_path, 'best.pt'), map_location=args.device)
        model.load_state_dict(checkpoint, strict=False)

    if args.do_train:
        if args.model == 'Pretrain':
            train_loader = load_data(args)
            pretrain(model, train_loader, args)
        else:
            train_loader, val_loader = load_data(args, ['train', 'val'])
            train(model, train_loader, val_loader, args)

            model = getattr(Model, args.model)(args).to(args.device)

            checkpoint = torch.load(os.path.join(args.output_path, 'best.pt'), map_location=args.device)
            model.load_state_dict(checkpoint)

    if args.do_test:
        test_loader = load_data(args, ['test'])
        test(model, test_loader, args)


if __name__ == '__main__':

    main()
