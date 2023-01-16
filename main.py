import torch
from tqdm import tqdm

from data import load_data
from loss import CELoss, SELoss
from config import get_config
from transformers import logging, AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt

import torch.nn.functional as F

from model import Transformer_CLS, Self_Attention, Self_Attention_New


def compare(predicta, predictb):
    # Softmax
    predict1 = F.softmax(predicta, dim=1)
    predict2 = F.softmax(predictb, dim=1)
    # Top2
    _, top_index = predict1.topk(2, 1, True, True)
    temp_index = 0
    for i in predict1:
        top1 = i[top_index[temp_index][0]]
        top2 = i[top_index[temp_index][1]]
        if (abs(top1 - top2) < 0.1):
            predict1[temp_index][top_index[temp_index][0]] = predict1[temp_index][top_index[temp_index][0]] + \
                                                             predict2[temp_index][top_index[temp_index][0]]
            predict1[temp_index][top_index[temp_index][1]] = predict1[temp_index][top_index[temp_index][1]] + \
                                                             predict2[temp_index][top_index[temp_index][1]]
        temp_index += 1
    return predict1


class Instructor:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info('> creating model {}'.format(args.model_name))
        if args.model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            base_model = AutoModel.from_pretrained('bert-base-uncased')
        elif args.model_name == 'wsp-large':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            base_model = AutoModel.from_pretrained("shuaifan/SentiWSP")
        elif args.model_name == 'wsp-base':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            base_model = AutoModel.from_pretrained("shuaifan/SentiWSP-base")
        else:
            raise ValueError('unknown model')

        if args.method_name == 'cls':
            self.model = Transformer_CLS(base_model, args.num_classes)
        elif args.method_name == 'sa':
            self.model = Self_Attention(base_model, args.num_classes)
        elif args.method_name == 'san':
            self.model = Self_Attention_New(base_model, args.num_classes)
        else:
            raise ValueError('unknown method')

        self.basic_model = Self_Attention_New(base_model, args.num_classes)

        self.model.to(args.device)
        self.basic_model.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _train(self, dataloader, criterion, optimizer, scheduler):
        train_loss, n_correct, n_train = 0, 0, 0

        self.model.train()
        for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            targets = targets.to(self.args.device)

            predicts = self.model(inputs)
            loss = criterion(predicts, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
            n_train += targets.size(0)

        return train_loss / n_train, n_correct / n_train

    def _test(self, dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        self.model.eval()

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)

                predicts = self.model(inputs)
                loss = criterion(predicts, targets)
                test_loss += loss.item() * targets.size(0)

                # predicts_base = self.basic_model(inputs)
                # predicts = compare(predicts, predicts_base)

                n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
                n_test += targets.size(0)

        return test_loss / n_test, n_correct / n_test

    def run(self):
        train_dataloader, test_dataloader = load_data(dataset=self.args.dataset,
                                                      data_dir=self.args.data_dir,
                                                      tokenizer=self.tokenizer,
                                                      train_batch_size=self.args.train_batch_size,
                                                      test_batch_size=self.args.test_batch_size,
                                                      model_name=self.args.model_name,
                                                      method_name=self.args.method_name,
                                                      workers=0)
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        criterion = CELoss()
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.decay)
        # state_dict = torch.load('./model.pkl')
        # self.basic_model.load_state_dict(state_dict)
        self.basic_model.eval()

        # Warm up
        total_steps = len(train_dataloader) * self.args.num_epoch
        warmup_steps = 0.1 * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)
        best_loss, best_acc = 0, 0

        l_acc, l_epo = [], []
        for epoch in range(self.args.num_epoch):
            train_loss, train_acc = self._train(train_dataloader, criterion, optimizer, scheduler)
            test_loss, test_acc = self._test(test_dataloader, criterion)
            l_epo.append(epoch), l_acc.append(test_acc)
            if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
                best_acc, best_loss = test_acc, test_loss
                # torch.save(self.model.state_dict(), './model.pkl')
            self.logger.info(
                '{}/{} - {:.2f}%'.format(epoch + 1, self.args.num_epoch, 100 * (epoch + 1) / self.args.num_epoch))
            self.logger.info('[train] loss: {:.4f}, acc: {:.2f}'.format(train_loss, train_acc * 100))
            self.logger.info('[test] loss: {:.4f}, acc: {:.2f}'.format(test_loss, test_acc * 100))
        self.logger.info('best loss: {:.4f}, best acc: {:.2f}'.format(best_loss, best_acc * 100))
        self.logger.info('log saved: {}'.format(self.args.log_name))

        # Draw the picture
        plt.plot(l_epo, l_acc)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig('image.png')
        plt.show()


if __name__ == '__main__':
    logging.set_verbosity_error()

    # 预设参数获取
    args, logger = get_config()

    # 将参数输入到模型中
    ins = Instructor(args, logger)

    # 模型训练评估
    ins.run()
