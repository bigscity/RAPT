import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as func
import models.transformer as transformer


class RAPT(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.fc = nn.Linear(args.input_dim, args.hidden_dim)

        self.position_encoding = transformer.PositionalEncoding(args.hidden_dim, args.max_len)
        self.cls = nn.Parameter(torch.randn(args.input_dim))

        self.transformers = nn.ModuleList(
            [transformer.TransformerEncoderLayer(input_dim=args.hidden_dim,
                                                 head_num=args.head_num,
                                                 hidden_dim=args.hidden_dim,
                                                 attention='TimeAwareMultiHeadAttention') for _ in range(args.layer_num)])

    def forward(self, x, mask, week, mode='pre'):

        if mode == 'pre':
            cls = self.cls.expand((x.size(0), 1, x.size(2)))

            x = torch.cat([cls, x], dim=1)

            last_index = torch.sum(mask, dim=-1).unsqueeze(1).long() - 1
            last_mask = torch.zeros_like(mask).scatter_(1, last_index, 1).long()
            last_week = week.masked_select(torch.eq(last_mask, 1)).view(-1, 1)

            week = torch.cat([last_week + 1, week], dim=1)
            mask = torch.cat([torch.ones(mask.size(0), 1, device=mask.device).long(), mask], dim=1)

        x = self.fc(x)
        p = self.position_encoding(week)

        for encoder in self.transformers:
            x = encoder(x, p, mask)

        if mode == 'pre':
            x = x[:, 0, :]

        return x


class Pretrain(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.task_func = dict(
            sp=self.similarity_prediction,
            mp=self.masked_prediction,
            rc=self.reasonability_check
        )

        self.args = args

        self.mode = args.mode
        self.task = args.task.split(',')
        self.proportion = [float(p) for p in args.proportion.split(',')]

        self.loss = ['loss_%s' % t for t in self.task] + ['loss']

        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim

        # masked prediction
        self.pre_n = args.pre_n
        self.mask_p = args.mask_p
        self.fc_mp = nn.Linear(self.hidden_dim, self.pre_n)

        # similarity prediction
        self.margin = args.margin

        # reasonability check
        self.fc_rc = nn.Linear(self.hidden_dim, 1)

        self.backbone = RAPT(args)

    def _proportion(self):
        if self.mode == 'random':
            select = np.random.choice(np.arange(0, len(self.task)), p=self.proportion)
            return [1 if i == select else 0 for i in range(len(self.task))]
        elif self.mode == 'stable':
            return self.proportion
        else:
            raise NotImplementedError

    def similarity_prediction(self, x, mask, week):

        last_index = torch.sum(mask, dim=-1).unsqueeze(1).long() - 1
        last_mask = torch.zeros_like(mask).scatter_(1, last_index, 1)
        last_x = x.masked_select(last_mask.unsqueeze(2).expand_as(x) == 1).view(x.size(0), x.size(2))[:, :4]
        dis = torch.sum(torch.pow(last_x[:last_x.size(0) // 2, :] - last_x[last_x.size(0) // 2:, :], 2), dim=1).cpu().numpy()

        y, m = list(), list()
        for d in dis:
            if d > 1089:
                y.append(0)
                m.append(1)
            elif d < 118:
                y.append(1)
                m.append(1)
            else:
                m.append(0)

        y = torch.tensor(y, device=self.args.device).float()
        m = torch.tensor(m, device=self.args.device)

        feature = self.backbone(x, mask, week, mode='pre')

        dis = torch.sum(torch.pow(feature[:feature.size(0) // 2, :] - feature[feature.size(0) // 2:, :], 2), dim=1)
        dis = dis.masked_select(torch.eq(m, 1))

        loss = torch.mean(y * dis + (1 - y) * torch.pow(self.margin - torch.sqrt(dis), 2))

        return loss

    def masked_prediction(self, x, mask, week):

        pre_mask = mask.masked_fill(torch.gt(torch.rand_like(mask), self.mask_p), 0).unsqueeze(2)
        y = x.masked_select(torch.eq(pre_mask.expand_as(x), 1)).view(-1, self.input_dim)[:, :self.pre_n]

        x = x.masked_fill(torch.eq(pre_mask.expand_as(x), 1), 0)
        m = self.backbone.cls.unsqueeze(0).unsqueeze(0).expand_as(x)
        m = m.masked_fill(torch.eq(pre_mask.expand_as(x), 0), 0)

        feature = self.backbone(x + m, mask, week, mode='s2s')
        feature = feature.masked_select(torch.eq(pre_mask.expand_as(feature), 1)).view(-1, self.args.hidden_dim)

        pred = self.fc_mp(feature)
        loss = func.mse_loss(pred, y)

        return loss

    def reasonability_check(self, x, mask, week):

        x, y = x.clone(), torch.zeros(x.size(0), device=self.args.device)
        for i in range(x.size(0)):
            if random.random() > 0.5:
                y[i] = 1

                l = int(torch.sum(mask[i]).item())

                sn = random.randint(l // 2, l)
                si = random.sample(list(range(l)), sn)
                sf = random.sample(list(range(x.size(0) * x.size(1))), sn)

                x[i, si, :] = x.view(-1, x.size(2))[sf, :]

        feature = self.backbone(x, mask, week, mode='pre')
        y_hat = torch.sigmoid(self.fc_rc(feature)).squeeze(1)
        loss = func.binary_cross_entropy(y_hat, y)

        return loss

    def forward(self, x, mask, week):

        proportion = self._proportion()
        result = dict(loss=torch.tensor(0.0, device=self.args.device))

        for task, p in zip(self.task, proportion):
            if p == 0:
                result['loss_%s' % task] = torch.tensor(0.0, device=self.args.device)
            else:
                result['loss_%s' % task] = self.task_func[task](x, mask, week)
                result['loss'] += p * result['loss_%s' % task]

        return result


class Predict(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.loss = ['loss']
        self.result = ['y_hat', 'y']

        self.backbone = RAPT(args)

        self.fc = nn.Linear(args.hidden_dim, 1)

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, x, mask, week, y):

        feature = self.backbone(x, mask, week)
        y_hat = torch.sigmoid(self.fc(self.dropout(feature)))

        y_hat, y = y_hat.squeeze(-1), y.squeeze(-1)
        loss = func.binary_cross_entropy(y_hat, y.float())

        loss = dict(loss=loss)
        result = dict(y=y.cpu().detach().numpy().tolist(),
                      feature=feature.cpu().detach().numpy(),
                      y_hat=y_hat.cpu().detach().numpy().tolist())

        return loss, result
