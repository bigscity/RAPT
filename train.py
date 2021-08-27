import os
import json
import torch
import logging
import numpy as np

from sklearn import metrics
from tensorboardX import SummaryWriter

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


def calculate_score(pred, label, scores, prefix=''):

    result = dict()
    for s in scores.split(','):
        if s == 'auc':
            result[prefix + s] = metrics.roc_auc_score(label, pred)
        elif s == 'f1':
            result['pre'] = 0
            result['rec'] = 0
            result[s] = 0
            for thresh in range(1, 10):
                prediction = [1 if p > (thresh / 10) else 0 for p in pred]
                f1 = 0 if len(prediction) == 0 else metrics.f1_score(label, prediction)
                if f1 > result[s]:
                    result['pre'] = metrics.precision_score(label, prediction)
                    result['rec'] = metrics.recall_score(label, prediction)
                    result[s] = f1
        elif s == 'acc':
            result[s] = 0
            for thresh in range(1, 10):
                prediction = [1 if p > (thresh / 10) else 0 for p in pred]
                acc = 0 if len(prediction) == 0 else metrics.accuracy_score(label, prediction)
                result[s] = max(result[s], acc)
        elif s == 'rmse':
            result[prefix + s] = metrics.mean_squared_error(label, pred, squared=False)
        elif s == 'mae':
            result[prefix + s] = metrics.mean_absolute_error(label, pred)
        elif s == 'mape':
            p, gt = np.array(pred), np.array(label)
            m = gt != 0
            result[prefix + s] = np.mean(np.fabs((p[m] - gt[m]) / gt[m]))
        elif s == 'r2':
            result[prefix + s] = metrics.r2_score(label, pred)
        elif s == 'ev':
            result[prefix + s] = metrics.explained_variance_score(label, pred)
        else:
            raise NotImplementedError

    return result


def pretrain(model, train_loaders, args):
    logging.info('Pretraining begin')

    writer = SummaryWriter(os.path.join(args.output_path, 'tensorboard'))

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.step_size, args.gamma)

    best_loss = float('inf')

    try:
        for epoch in range(args.epoch_num):
            model.train()

            train_loss = {name: list() for name in model.loss}

            for batch in train_loaders:
                batch = batch.to(device=args.device)
                loss = model(batch.x, batch.mask, batch.week)

                optimizer.zero_grad()
                loss['loss'].backward()
                optimizer.step()

                for k in train_loss:
                    train_loss[k].append(loss[k].item())

            for k in train_loss:
                train_loss[k] = 0 if len(train_loss[k]) == 0 else sum(train_loss[k]) / len(train_loss[k])
                writer.add_scalar(k, train_loss[k], epoch)

            best = ''
            if train_loss['loss'] <= best_loss:
                best_loss, best = train_loss['loss'], '*'
                torch.save(model.state_dict(), os.path.join(args.output_path, 'best.pt'))

            train_loss_msg = ' '.join('{}:{: .6f}'.format(k, v) for k, v in train_loss.items())
            logging.info('Epoch: [{}/{}] lr: {:.2g} {} {}'.format(epoch + 1, args.epoch_num, optimizer.param_groups[0]['lr'], train_loss_msg, best))

            scheduler.step()

    except KeyboardInterrupt:
        logging.info('KeyboardInterrupt')
    except Exception as e:
        logging.exception(e)

    writer.close()
    logging.info('Pretraining end')


def validate(model, val_loader, args):

    model.eval()

    with torch.no_grad():

        val_loss = {name: list() for name in model.loss}
        val_result = {name: list() for name in model.result}

        for batch in val_loader:
            batch = batch.to(device=args.device)

            loss, result = model(batch.x, batch.mask, batch.week, batch.y)

            for k in val_loss:
                val_loss[k].append(loss[k].item())

            for k in val_result:
                val_result[k].extend(result[k])

        val_score = calculate_score(val_result['y_hat'], val_result['y'], args.scores)

        for k in val_loss:
            val_loss[k] = sum(val_loss[k]) / len(val_loss[k])

    return val_loss, val_score


def train(model, train_loader, val_loader, args):

    writer = SummaryWriter(os.path.join(args.output_path, 'tensorboard'))

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.step_size, args.gamma)

    best_score = float('-inf')
    msg = 'Epoch: [{}/%d] lr: {:.2g} \n\t train {} {} val {} {} {}' % args.epoch_num

    logging.info('Training begin')
    try:
        for epoch in range(1, args.epoch_num + 1):

            model.train()

            train_loss = {name: list() for name in model.loss}
            train_result = {name: list() for name in model.result}

            for batch in train_loader:
                batch = batch.to(device=args.device)

                loss, result = model(batch.x, batch.mask, batch.week, batch.y)

                optimizer.zero_grad()
                loss['loss'].backward()
                optimizer.step()

                for k in train_loss:
                    train_loss[k].append(loss[k].item())

                for k in train_result:
                    train_result[k].extend(result[k])

            train_score = calculate_score(train_result['y_hat'], train_result['y'], args.scores)

            for k in train_loss:
                train_loss[k] = sum(train_loss[k]) / len(train_loss[k])

            val_loss, val_score = validate(model, val_loader, args)

            for loss in train_loss:
                writer.add_scalars(f'loss/{loss}', {'train': train_loss[loss], 'val': val_loss[loss]}, epoch)
            for score in train_score:
                writer.add_scalars(f'score/{score}', {'train': train_score[score], 'val': val_score[score]}, epoch)

            best, main_score = '', val_score[args.main_score]

            if main_score >= best_score:
                best_score, best = main_score, '*'
                torch.save(model.state_dict(), os.path.join(args.output_path, 'best.pt'))

            train_loss_msg = ' '.join('{}:{: .3f}'.format(k, v) for k, v in train_loss.items())
            val_loss_msg = ' '.join('{}:{: .3f}'.format(k, v) for k, v in val_loss.items())

            train_score_msg = ' '.join(['{}:{: .3f}'.format(k, v) for k, v in train_score.items()])
            val_score_msg = ' '.join(['{}:{: .3f}'.format(k, v) for k, v in val_score.items()])

            logging.info(msg.format(epoch, optimizer.param_groups[0]['lr'], train_loss_msg, train_score_msg, val_loss_msg, val_score_msg, best))

            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            scheduler.step()

    except KeyboardInterrupt:
        logging.info('KeyboardInterrupt best val score: {:.3f}'.format(best_score))
    except Exception as e:
        logging.exception(e)

    writer.close()

    logging.info('Training end best val score: {:.3f}'.format(best_score))


def test(model, test_loader, args):

    model.eval()

    with torch.no_grad():

        test_loss = {name: list() for name in model.loss}
        test_result = {name: list() for name in model.result}

        for batch in test_loader:
            batch = batch.to(device=args.device)
            loss, result = model(batch.x, batch.mask, batch.week, batch.y)

            for k in test_loss:
                test_loss[k].append(loss[k].item())

            for k in test_result:
                test_result[k].extend(result[k])

        test_score = calculate_score(test_result['y_hat'], test_result['y'], args.scores)

        for k in test_loss:
            test_loss[k] = sum(test_loss[k]) / len(test_loss[k])

        test_loss_msg = ' '.join('{}:{: .3f}'.format(k, v) for k, v in test_loss.items())
        test_score_msg = ' '.join(['{}:{: .3f}'.format(k, v) for k, v in test_score.items()])

        logging.info('Test {} {}'.format(test_loss_msg, test_score_msg))

        test_json = dict(loss=test_loss, score=test_score)
        with open(os.path.join(args.output_path, 'test.json'), 'w') as fw:
            json.dump(test_json, fw)
