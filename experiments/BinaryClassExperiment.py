import os
import pickle

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from datasets.three_dim.NumpyDataLoader import NumpyDataSet
from trixi.experiment.pytorchexperiment import PytorchExperiment

from networks.ClassificationNN import ClassificationNN

from loss_functions.dice_loss import SoftDiceLoss

from datasets.chd_dataset.load_excel import load_excel



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

        self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')    #

        self.train_data_loader = NumpyDataSet(self.config.data_dir, target_size=(128, 128, 128), batch_size=self.config.batch_size,
                                              keys=tr_keys, do_reshuffle=True)
        self.val_data_loader = NumpyDataSet(self.config.data_dir, target_size=(128, 128, 128), batch_size=self.config.batch_size,
                                            keys=val_keys, mode="val", do_reshuffle=True)
        self.test_data_loader = NumpyDataSet(self.config.data_dir, target_size=(128, 128, 128), batch_size=self.config.batch_size,
                                             keys=test_keys, mode="test", do_reshuffle=False)
        self.model = ClassificationNN()

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)


        # We use a combination of DICE-loss and CE-Loss in this example.
        # This proved good in the medical segmentation decathlon.
        self.dice_loss = SoftDiceLoss(batch_dice=True)  # Softmax für DICE Loss!

        # weight = torch.tensor([1, 30, 30]).float().to(self.device)
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
        for data_batch in self.train_data_loader:

            self.optimizer.zero_grad()

            target = []

            # Shape of data_batch = [1, b, c, w, h]
            # Desired shape = [b, c, w, h]
            # Move data and target to the GPU
            data = data_batch['data'][0].float().to(self.device)
            tapvc_dict = load_excel(self.config.excel_dir)
            fname_list = data_batch['fnames']
            for fname in fname_list:
                file = fname[0].split('preprocessed/')[1]
                assert file in tapvc_dict, 'number of .npy is not in pvo excel'
                target.append(tapvc_dict[file])

            target = torch.FloatTensor(target).to(self.device).long()

            # target = data_batch['seg'][0].long().to(self.device)

            pred = self.model(data.squeeze()) # should be of size (N, 2)

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

        data = None
        loss_list = []
        num_pvo = 0
        correct_pvo = 0

        with torch.no_grad():
            for data_batch in self.val_data_loader:
                data = data_batch['data'][0].float().to(self.device)

                target = []
                tapvc_dict = load_excel(self.config.excel_dir)
                fname_list = data_batch['fnames']
                for fname in fname_list:
                    file = fname[0].split('preprocessed/')[1]
                    assert file in tapvc_dict, 'number of .npy is not in pvo excel'
                    target.append(tapvc_dict[file])

                target = torch.FloatTensor(target).to(self.device).long()

                pred = self.model(data.squeeze())

                pred_softmax = F.softmax(pred)
                pred_pvo = torch.argmax(pred_softmax, dim=1)

                c = pred_pvo.mul(target)

                correct_pvo += sum(c)
                num_pvo += sum(target)


                loss = self.ce_loss(pred, target.squeeze())
                loss_list.append(loss.item())

        assert data is not None, 'data is None. Please check if your dataloader works properly'
        self.scheduler.step(np.mean(loss_list))

        self.elog.print('Epoch: %d Loss: %.4f Accuracy: %.4f' % (self._epoch_idx, np.mean(loss_list), correct_pvo/num_pvo))

        self.add_result(value=np.mean(loss_list), name='Val_Loss', tag='Loss', counter=epoch+1)

        # self.clog.show_image_grid(data.float(), name="data_val", normalize=True, scale_each=True, n_iter=epoch)
        # self.clog.show_image_grid(target.float(), name="mask_val", title="Mask", n_iter=epoch)
        # self.clog.show_image_grid(torch.argmax(pred.data.cpu(), dim=1, keepdim=True), name="unt_argmax_val", title="Unet", n_iter=epoch)
        # self.clog.show_image_grid(pred.data.cpu()[:, 1:2, ], name="unt_val", normalize=True, scale_each=True, n_iter=epoch)

    def test(self):

        self.model.eval()
        data = None

        num_of_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("number of parameters:", num_of_parameters)

        segmented_dir = os.path.join(self.config.split_dir, 'segmented')
        if not os.path.exists(segmented_dir):
            os.mkdir(segmented_dir)

        with torch.no_grad():
            for data_batch in self.test_data_loader:
                data = data_batch['data'][0].float().to(self.device) # shape(N, 1, d, d, d)
                file_dir = data_batch['fnames']  # 8*tuple (a,)

                pred = self.model(data.squeeze())
                # pred_softmax = F.softmax(pred, dim=1)  # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.
                # pred_image = torch.argmax(pred_softmax, dim=1)

                for k in range(self.config.batch_size):
                    # save the results
                    # pred = pred_softmax[k].reshape((3,64,64))
                    filename = file_dir[k][0]
                    output_dir = os.path.join(segmented_dir, 'segmented_' + filename)

                    if os.path.exists(output_dir + '.npy'):
                        all_image = np.load(output_dir + '.npy')
                        output = pred[k]
                        all_image = np.concatenate((all_image, output), axis=0)
                    else:
                        all_image = pred[k]

                    print(output_dir)
                    np.save(output_dir, all_image)
                #    saveName = filenames[k]


            print('test_data loading finished')

        assert data is not None, 'data is None. Please check if your dataloader works properly'

        # print('TODO: Implement your test() method here')
