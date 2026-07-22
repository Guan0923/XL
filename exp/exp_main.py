from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import FreDF_XLinear, XLinear, XLinear_FFT, XLinear_FFT_X, XLinear_FFT_Fre

try:
    from models import XLinear_ES
except ImportError:
    XLinear_ES = None

try:
    from models import XLinear_GT
except ImportError:
    XLinear_GT = None
from utils.tools import (EarlyStopping, LossGradientFeedbackLRController,
                         PlateauLRController, adjust_learning_rate, visual,
                         test_params_flop)
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'XLinear':XLinear,
            'XLinear-FFT':XLinear_FFT,
            'XLinear-FFT-X':XLinear_FFT_X,
            'FreDF-XLinear':FreDF_XLinear,
            'XLinear-FFT-Fre':XLinear_FFT_Fre
        }
        if XLinear_ES is not None:
            model_dict['XLinear-ES'] = XLinear_ES
        if XLinear_GT is not None:
            model_dict['XLinear-GT'] = XLinear_GT
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def _save_fft_heatmaps(self, setting, epoch):
        """保存 XLinear-FFT backbone 中 batch 0 的输入(FFT前)与 IFFT 输出(FFT后)折线图(PDF)。
        每个 epoch 生成一个 PDF，PDF 内每个通道一个子图，子图内同时画 input 和 irfft 两条曲线。
        仅在每个 epoch 的最后一个 iter 调用。"""
        if self.args.model != 'XLinear-FFT':
            return
        model = self.model.module if hasattr(self.model, 'module') else self.model
        backbone = getattr(model, 'backbone', None)
        if backbone is None:
            return
        folder_path = './test_results/' + setting + '/heatmaps/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        inp = getattr(backbone, 'input_b0', None)
        irf = getattr(backbone, 'irfft_b0', None)
        if inp is None and irf is None:
            return

        inp_arr = inp.detach().cpu().numpy() if inp is not None else None
        irf_arr = irf.detach().cpu().numpy() if irf is not None else None

        # 通道数取存在的那一方
        if inp_arr is not None:
            C = inp_arr.shape[0]
        else:
            C = irf_arr.shape[0]

        fig, axes = plt.subplots(C, 1, figsize=(10, 2 * C), sharex=False)
        if C == 1:
            axes = [axes]

        for c in range(C):
            if inp_arr is not None:
                x_in = np.linspace(0, 1, inp_arr.shape[1])
                axes[c].plot(x_in, inp_arr[c], linewidth=0.9,
                             color='tab:blue', label='input(FFT前)')
            if irf_arr is not None:
                x_ir = np.linspace(0, 1, irf_arr.shape[1])
                axes[c].plot(x_ir, irf_arr[c], linewidth=0.9,
                             color='tab:red', label='irfft(FFT后)')
            axes[c].set_ylabel('ch{}'.format(c))
            axes[c].grid(True, linestyle='--', alpha=0.4)
            axes[c].legend(loc='best', fontsize=7)

        axes[-1].set_xlabel('Relative Time (0~1)')
        fig.suptitle('batch0 FFT before/after (epoch {})'.format(epoch + 1))
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(os.path.join(folder_path,
                                 'epoch{}.pdf'.format(epoch + 1)))
        plt.close(fig)

    def _save_glob_heatmaps(self, setting, epoch):
        """保存 glob_token 的灰度热力图（实部 + 虚部）。
        每个 epoch 生成一个 PNG，x=频率bin，y=通道。"""
        if self.args.model != 'XLinear-FFT-X':
            return
        model = self.model.module if hasattr(self.model, 'module') else self.model
        backbone = getattr(model, 'backbone', None)
        if backbone is None:
            return
        # glob_token 仅当 glob_dim > 0 时存在
        if not hasattr(backbone, 'glob_token_real') or backbone.glob_token_real is None:
            return

        folder_path = './test_results/' + setting + '/glob_heatmaps/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        real_arr = backbone.glob_token_real.detach().cpu().numpy()[0]  # [C, glob_dim]
        imag_arr = backbone.glob_token_imag.detach().cpu().numpy()[0]  # [C, glob_dim]

        fig, axes = plt.subplots(2, 1, figsize=(10, 4))

        vmax = max(abs(real_arr).max(), abs(imag_arr).max()) + 1e-8

        im0 = axes[0].imshow(real_arr, aspect='auto', cmap='gray',
                             vmin=-vmax, vmax=vmax, interpolation='nearest')
        axes[0].set_title('glob_token real (epoch {})'.format(epoch + 1))
        axes[0].set_xlabel('frequency bin (high-freq replaced)')
        axes[0].set_ylabel('channel')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(imag_arr, aspect='auto', cmap='gray',
                             vmin=-vmax, vmax=vmax, interpolation='nearest')
        axes[1].set_title('glob_token imag (epoch {})'.format(epoch + 1))
        axes[1].set_xlabel('frequency bin (high-freq replaced)')
        axes[1].set_ylabel('channel')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(os.path.join(folder_path, 'epoch{}.png'.format(epoch + 1)))
        plt.close(fig)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scaler = None
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        lr_controller = None
        if self.args.use_lgflr:
            lgf_mode = getattr(self.args, 'lgf_mode', 'full')
            if lgf_mode == 'plateau':
                lr_controller = PlateauLRController(
                    optimizer=model_optim,
                    patience=self.args.lgf_plateau_patience,
                    factor=self.args.lgf_plateau_factor,
                    eta_min=self.args.lgf_plateau_eta_min)
                print('Using plateau loss learning-rate controller')
            else:
                lr_controller = LossGradientFeedbackLRController(
                    model=self.model,
                    optimizer=model_optim,
                    checkpoint_path=os.path.join(path, 'lgflr_trial.pth'),
                    scaler=scaler,
                    beta=self.args.lgf_beta,
                    beta_g=self.args.lgf_beta_g,
                    tau_down=self.args.lgf_tau_down,
                    tau_up=self.args.lgf_tau_up,
                    p_good=self.args.lgf_p_good,
                    p_bad=self.args.lgf_p_bad,
                    t_rec=self.args.lgf_t_rec,
                    t_trial=self.args.lgf_t_trial,
                    gamma_good=self.args.lgf_gamma_good,
                    gamma_down=self.args.lgf_gamma_down,
                    gamma_up=self.args.lgf_gamma_up,
                    gamma_safe=self.args.lgf_gamma_safe,
                    tau_rec=self.args.lgf_tau_rec,
                    tau_accept=self.args.lgf_tau_accept,
                    grad_window=self.args.lgf_grad_window,
                    kappa_g=self.args.lgf_kappa_g,
                    epsilon=self.args.lgf_epsilon,
                    eta_min=self.args.lgf_eta_min,
                    eta_max=self.args.lgf_eta_max)
                print('Using loss-gradient feedback learning-rate controller')

        scheduler = None
        if not self.args.use_lgflr and self.args.lradj == 'TST':
            scheduler = lr_scheduler.OneCycleLR(
                optimizer=model_optim,
                steps_per_epoch=train_steps,
                pct_start=self.args.pct_start,
                epochs=self.args.train_epochs,
                max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_gradient_norm = 0.0

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark)
                    # print(outputs.shape,batch_y.shape)
                    
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    if lr_controller is not None and lr_controller.requires_gradient:
                        scaler.unscale_(model_optim)
                        epoch_gradient_norm += lr_controller.compute_gradient_norm(
                            self.model.parameters())
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    if lr_controller is not None and lr_controller.requires_gradient:
                        epoch_gradient_norm += lr_controller.compute_gradient_norm(
                            self.model.parameters())
                    model_optim.step()
                    
            if not self.args.use_lgflr and self.args.lradj == 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                scheduler.step()

            # 每个 epoch 最后一个 iter：保存 batch 0 的灰度热力图 + glob_token 热力图
            if i == train_steps - 1:
                self._save_fft_heatmaps(setting, epoch)
                self._save_glob_heatmaps(setting, epoch)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            # ---- glob_token 可视化（仅 XLinear-FFT-X 模型） ----
            if self.args.model == 'XLinear-FFT-X' and hasattr(self.model, 'backbone'):
                bb = self.model.backbone
                if hasattr(bb, '_cached_glob_stats') and bb._cached_glob_stats is not None:
                    r_mean, r_std, i_mean, i_std, r_max, i_max = bb._cached_glob_stats
                    print("  [glob_token] real_mean={:.6f} real_std={:.6f} imag_mean={:.6f} imag_std={:.6f} | real_max={:.6f} imag_max={:.6f}".format(
                        r_mean, r_std, i_mean, i_std, r_max, i_max))
                    bb._cached_glob_stats = None  # 重置，下一个 epoch 重新缓存
            # 早停改用 test loss（仅供观察模型最优，存在 test 泄漏，不建议正式实验使用）
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if lr_controller is not None:
                if lr_controller.requires_gradient:
                    controller_result = lr_controller.step(
                        validation_loss=vali_loss,
                        train_epoch_loss=train_loss,
                        epoch_grad_norm=epoch_gradient_norm / train_steps)
                    print(
                        'LGF-LR | state: {state}, event: {event}, lr: {lr:.8g}, '
                        'r_loss: {r_loss}, q_grad: {q_grad:.6f}'.format(
                            state=controller_result['state'],
                            event=controller_result['event'],
                            lr=controller_result['learning_rate'],
                            r_loss=('N/A' if controller_result['relative_loss_change'] is None
                                    else '{:.6f}'.format(
                                        controller_result['relative_loss_change'])),
                            q_grad=controller_result['relative_gradient']))
                else:
                    controller_result = lr_controller.step(
                        validation_loss=vali_loss,
                        train_epoch_loss=train_loss)
                    print(
                        'Plateau-LR | event: {event}, lr: {lr:.8g}, '
                        'best: {best:.7f}, bad_epochs: {bad}/{patience}'.format(
                            event=controller_result['event'],
                            lr=controller_result['learning_rate'],
                            best=controller_result['best_loss'],
                            bad=controller_result['bad_epochs'],
                            patience=controller_result['patience']))
            elif self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        if lr_controller is not None:
            lr_controller.close()

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        print("find shape", preds.shape, trues.shape)
        mae, mse, rmse, mape, mspe, rse, corr, nse, kge, r2 = metric(preds, trues)
        print('mse:{}, mae:{}, nse:{}, kge:{}, mape:{}'.format(mse, mae, nse, kge, mape))
        f = open(f"result_{setting}.txt", 'w')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
