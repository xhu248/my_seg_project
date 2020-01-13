import os
import pickle
import csv

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from datasets.three_dim.NumpyDataLoader import NumpyDataSet
from trixi.experiment.pytorchexperiment import PytorchExperiment

from networks.ClassificationNN import ClassificationNN
from networks.ClassificationNN import ClassificationUnet
from networks.ClassificationRes import ClassificationVnet

from loss_functions.dice_loss import SoftDiceLoss

from datasets.tapvc_dataset.load_excel import load_excel

from utilities.metrics import print_metrices_out



class BinaryClassExperiment(PytorchExperiment):
    """
    The UnetExperiment is inherited from the PytorchExperiment. It implements the basic life cycle for a segmentation task with UNet(https://arxiv.org/abs/1505.04597).
    It is optimized to work with the provided NumpyDataLoader.

    The basic life cycle of a UnetExperiment is the same s PytorchExperiment:

        setup()
        (--> Automatically restore values if a previous checkpoint is given)
        prepare()

        for epoch in n_epochs:
            train()
            validate()
            (--> save current checkpoint)

        end()
    """

    def setup(self):

        pkl_dir = self.config.split_dir
        with open(os.path.join(pkl_dir, "splits.pkl"), 'rb') as f:
            splits = pickle.load(f)

        tr_keys = splits[self.config.fold]['train']
        val_keys = splits[self.config.fold]['val']
        test_keys = splits[self.config.fold]['test']
        all_keys = splits[self.config.fold]['all']

        val_keys = val_keys + test_keys

        self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')    #

        self.train_data_loader = NumpyDataSet(self.config.data_dir, target_size=(128, 128, 128), batch_size=self.config.batch_size,
                                              keys=tr_keys, do_reshuffle=True)
        self.val_data_loader = NumpyDataSet(self.config.data_dir, target_size=(128, 128, 128), batch_size=self.config.batch_size,
                                            keys=val_keys, mode="val", do_reshuffle=True)
        self.test_data_loader = NumpyDataSet(self.config.data_dir, target_size=(128, 128, 128), batch_size=1,
                                             keys=val_keys, mode="test", do_reshuffle=False)
        # self.model = ClassificationNN()
        self.model = ClassificationUnet(initial_filter_size=32, num_downs=3)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)


        # We use a combination of DICE-loss and CE-Loss in this example.
        # This proved good in the medical segmentation decathlon.
        self.dice_loss = SoftDiceLoss(batch_dice=True)  # Softmax für DICE Loss!

        weight = torch.FloatTensor([1, 20]).to(self.device)
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=weight)  # Kein Softmax für CE Loss -> ist in torch schon mit drin!
        # self.dice_pytorch = dice_pytorch(self.config.num_classes)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate)

        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        # If directory for checkpoint is provided, we load it.
        if self.config.do_load_checkpoint:
            if self.config.checkpoint_dir == '':
                print('checkpoint_dir is empty, please provide directory to load checkpoint.')
            else:
                self.load_checkpoint(name=self.config.checkpoint_dir, save_types=("model"))

        self.save_checkpoint(name="checkpoint_start")
        self.elog.print('Experiment set up.')

    def train(self, epoch):
        self.elog.print('=====TRAIN=====')
        self.model.train()

        data = None
        batch_counter = 0

        tapvc_dict = load_excel(self.config.excel_dir, do_random=False)
        for data_batch in self.train_data_loader:

            self.optimizer.zero_grad()

            target = []

            # Shape of data_batch = [1, b, c, w, h]
            # Desired shape = [b, c, w, h]
            # Move data and target to the GPU
            data = data_batch['data'][0].float().to(self.device)
            fname_list = data_batch['fnames']
            for fname in fname_list:
                file = fname[0].split('preprocessed/')[1]
                assert file in tapvc_dict, 'number of .npy is not in pvo excel'
                target.append(tapvc_dict[file])

            target = torch.FloatTensor(target).to(self.device).long()

            # target = data_batch['seg'][0].long().to(self.device)

            _, pred = self.model(data)  # treating data as 3d image
            # pred = self.model(data.squeeze()) # should be of size (N, 2)

            loss = self.ce_loss(pred, target.squeeze())
            # loss = self.dice_loss(pred_softmax, target.squeeze())
            loss.backward()
            self.optimizer.step()

            # Some logging and plotting
            if (batch_counter % self.config.plot_freq) == 0:
                self.elog.print('Epoch: %d Loss: %.4f' % (self._epoch_idx, loss))

                self.add_result(value=loss.item(), name='Train_Loss', tag='Loss', counter=epoch + (batch_counter / self.train_data_loader.data_loader.num_batches))

                # self.clog.show_image_grid(data.float(), name="data", normalize=True, scale_each=True, n_iter=epoch)
                # self.clog.show_image_grid(target.float(), name="mask", title="Mask", n_iter=epoch)
                # self.clog.show_image_grid(torch.argmax(pred.cpu(), dim=1, keepdim=True), name="unt_argmax", title="Unet", n_iter=epoch)
                # self.clog.show_image_grid(pred.cpu()[:, 1:2, ], name="unt", normalize=True, scale_each=True, n_iter=epoch)

            batch_counter += 1

        assert data is not None, 'data is None. Please check if your dataloader works properly'

    def validate(self, epoch):
        self.elog.print('VALIDATE')
        self.model.eval()

        tapvc_dict = load_excel(self.config.excel_dir, do_random=False)
        data = None
        loss_list = []
        total_num = 0
        num_pvo = 0
        correct_pvo = 0
        false_alarm = 0

        with torch.no_grad():
            for data_batch in self.val_data_loader:
                data = data_batch['data'][0].float().to(self.device)

                target = []
                fname_list = data_batch['fnames']
                for fname in fname_list:
                    file = fname[0].split('preprocessed/')[1]
                    assert file in tapvc_dict, 'number of .npy is not in pvo excel'
                    target.append(tapvc_dict[file])

                target = torch.FloatTensor(target).to(self.device).long()

                features, pred = self.model(data)

                pred_softmax = F.softmax(pred, dim=1)
                pred_pvo = torch.argmax(pred_softmax, dim=1)

                c = pred_pvo.mul(target)

                total_num += sum(pred_pvo)
                correct_pvo += sum(c)
                num_pvo += sum(target)

                loss = self.ce_loss(pred, target.squeeze())
                loss_list.append(loss.item())

        wrong_pvo_num = total_num - correct_pvo
        accuracy = correct_pvo.item() / num_pvo.item()
        assert data is not None, 'data is None. Please check if your dataloader works properly'
        self.scheduler.step(np.mean(loss_list))

        self.elog.print('Epoch: %d Loss: %.4f Accuracy: %.4f NUmber of wrong pvo: %d' % (self._epoch_idx, np.mean(loss_list), accuracy, wrong_pvo_num))

        self.add_result(value=np.mean(loss_list), name='Val_Loss', tag='Loss', counter=epoch+1)
        self.add_result(value=accuracy, name='val_accuracy', counter=epoch+1)
        self.add_result(value=wrong_pvo_num.item(), name='val_fales', counter=epoch+1)

        # self.clog.show_image_grid(data.float(), name="data_val", normalize=True, scale_each=True, n_iter=epoch)
        # self.clog.show_image_grid(target.float(), name="mask_val", title="Mask", n_iter=epoch)
        # self.clog.show_image_grid(torch.argmax(pred.data.cpu(), dim=1, keepdim=True), name="unt_argmax_val", title="Unet", n_iter=epoch)
        # self.clog.show_image_grid(pred.data.cpu()[:, 1:2, ], name="unt_val", normalize=True, scale_each=True, n_iter=epoch)

    def test(self):
        self.model.eval()
        data = None

        tapvc_dict = load_excel(self.config.excel_dir, do_random=False)

        feature_dict = {}
        correct_list = []

        # y_predict,  y_test and y_prob are used to print metrics
        y_predict = []
        y_test = []
        y_prob = []

        with torch.no_grad():
            for data_batch in self.test_data_loader:
                data = data_batch['data'][0].float().to(self.device) # shape(N, 1, d, d, d)
                fname_list = data_batch['fnames']  # 8*tuple (a,)

                fname = fname_list[0][0].split('preprocessed/')[1]
                assert fname in tapvc_dict, 'number of .npy is not in pvo excel'
                target = tapvc_dict[fname]

                features, pred = self.model(data)
                pred_softmax = F.softmax(pred, dim=1)  # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.
                # pred_image = torch.argmax(pred_softmax, dim=1)

                pred_softmax = F.softmax(pred, dim=1)
                pred_pvo = torch.argmax(pred_softmax, dim=1)

                y_predict.append(pred_pvo.cpu().numpy())
                y_test.append(target)
                y_prob.append(pred_softmax[0][1].cpu().numpy())

                if pred_pvo == 1 and target == 1:
                        correct_list.append(fname)

                # store features
                new_num = fname.split('.')[0]
                features = features.squeeze()
                features = features.cpu().numpy()
                features = list(features)
                feature_dict[new_num] = features

                print("Patient's new number:", new_num)
                print("pvo ground truth: %d, pvo prediction: %d" % (target, pred_pvo))

        with open("feature_dict.pkl", 'wb') as f:
            pickle.dump(feature_dict, f)

        y_predict = np.array(y_predict)
        y_test = np.array(y_test)
        y_prob = np.array(y_prob)
        print_metrices_out(y_predict, y_test, y_prob)
        print(correct_list)

        assert data is not None, 'data is None. Please check if your dataloader works properly'

        # print('TODO: Implement your test() method here')
