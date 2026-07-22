import math
import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import time

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class LossGradientFeedbackLRController:
    """Loss-gradient learning-rate control with validation and rollback.

    The controller uses the gradient only to propose a bounded exploration.
    Smoothed loss validates the proposal, and a full checkpoint corrects a
    failed proposal. The default values are heuristic starting points and
    should be verified by ablation and sensitivity experiments.

    Call ``step`` once per epoch. ``train_epoch_loss`` must be the mean over
    the whole epoch rather than the loss of its final batch. If AMP is used,
    call ``scaler.unscale_(optimizer)`` before ``compute_gradient_norm`` and
    average the returned batch norms over the epoch.

    参数说明：
        model: 被训练的模型。探索前保存其参数，探索失败时完整恢复。
        optimizer: 被控制的优化器。控制器修改学习率，并在回滚时恢复
            参数组、动量等完整优化器状态。
        checkpoint_path: 探索临时 checkpoint 的路径，与 EarlyStopping
            保存的最佳模型 checkpoint 相互独立。
        scaler: 可选 AMP GradScaler，提供时随 checkpoint 保存和恢复。
        beta: loss 的 EMA 系数。越接近 1 越平滑、响应越慢；
            tau_down 和 tau_up 均作用于平滑后的相对 loss 变化率。
        beta_g: 梯度范数的 EMA 系数，越接近 1 越平滑。
        tau_down: 有效下降阈值，r_t < -tau_down 时计为有效下降。
        tau_up: 有效上升阈值，r_t > tau_up 时计为有效上升；
            两个阈值之间的变化被视为 stable。
        p_good: 触发 gamma_good 缓慢衰减所需的连续下降次数。
        p_bad: 容许的连续上升次数。实现使用 bad_count > p_bad，
            因此默认 2 表示容忍两次、第三次触发恢复。
        t_rec: 降低学习率后的固定恢复观察 epoch 数。
        t_trial: 提高学习率后的固定探索观察 epoch 数。
        gamma_good: 连续下降时的温和学习率衰减系数。
        gamma_down: 确认不稳定或高梯度恢复失败后的衰减系数。
        gamma_up: 低相对梯度提出探索时的学习率放大系数。
        gamma_safe: 探索失败并回滚后的安全衰减系数。
        tau_rec: 接受恢复结果所需的最小相对 loss 改善率。
        tau_accept: 接受探索结果所需的最小相对 loss 改善率。
        grad_window: 计算近期平滑梯度中位数参考值的 epoch 数 M。
        kappa_g: 相对梯度阈值，仅在恢复失败后检查；q_t^g 小于该值
            时梯度提议进入探索。
        epsilon: loss 相对变化率和梯度比值分母的防零常数。
        eta_min: 控制器允许设置的最小学习率。
        eta_max: 控制器允许设置的最大学习率；None 表示使用初始
            学习率，因此探索默认不会超过初始学习率。

    所有默认值均为启发式初值，不代表理论最优值，需要通过消融实验
    和敏感性分析针对具体模型与数据集进行验证。
    """

    NORMAL = 'normal'
    RECOVERY = 'recovery'
    TRIAL = 'trial'
    ROLLBACK = 'rollback'
    _STATE_DICT_VERSION = 1
    requires_gradient = True

    def __init__(
            self,
            model,
            optimizer,
            checkpoint_path,
            scaler=None,
            beta=0.9,
            beta_g=0.9,
            tau_down=0.002,
            tau_up=0.005,
            p_good=2,
            p_bad=2,
            t_rec=3,
            t_trial=2,
            gamma_good=0.98,
            gamma_down=0.5,
            gamma_up=1.25,
            gamma_safe=0.5,
            tau_rec=0.002,
            tau_accept=0.002,
            grad_window=5,
            kappa_g=0.5,
            epsilon=1e-12,
            eta_min=1e-7,
            eta_max=None):
        if not optimizer.param_groups:
            raise ValueError('optimizer must contain at least one parameter group')

        initial_lr = float(optimizer.param_groups[0]['lr'])
        if eta_max is None:
            eta_max = initial_lr

        self._validate_hyperparameters(
            beta=beta,
            beta_g=beta_g,
            tau_down=tau_down,
            tau_up=tau_up,
            p_good=p_good,
            p_bad=p_bad,
            t_rec=t_rec,
            t_trial=t_trial,
            gamma_good=gamma_good,
            gamma_down=gamma_down,
            gamma_up=gamma_up,
            gamma_safe=gamma_safe,
            tau_rec=tau_rec,
            tau_accept=tau_accept,
            grad_window=grad_window,
            kappa_g=kappa_g,
            epsilon=epsilon,
            eta_min=eta_min,
            eta_max=eta_max)

        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.checkpoint_path = os.fspath(checkpoint_path)

        self.beta = float(beta)
        self.beta_g = float(beta_g)
        self.tau_down = float(tau_down)
        self.tau_up = float(tau_up)
        self.p_good = int(p_good)
        self.p_bad = int(p_bad)
        self.t_rec = int(t_rec)
        self.t_trial = int(t_trial)
        self.gamma_good = float(gamma_good)
        self.gamma_down = float(gamma_down)
        self.gamma_up = float(gamma_up)
        self.gamma_safe = float(gamma_safe)
        self.tau_rec = float(tau_rec)
        self.tau_accept = float(tau_accept)
        self.grad_window = int(grad_window)
        self.kappa_g = float(kappa_g)
        self.epsilon = float(epsilon)
        self.eta_min = float(eta_min)
        self.eta_max = float(eta_max)

        self.state = self.NORMAL
        self.steps = 0
        self.smoothed_loss = None
        self.smoothed_grad_norm = None
        self.gradient_history = []
        self.good_count = 0
        self.bad_count = 0

        self.base_loss = None
        self.pre_reduction_lr = None
        self.recovery_lr = None
        self.recovery_count = 0
        self.recovery_best_loss = None

        self.trial_count = 0
        self.trial_best_loss = None
        self.saved_lr = None

        self._set_lr(initial_lr)

    @staticmethod
    def _validate_hyperparameters(**values):
        for name in ('beta', 'beta_g'):
            if not 0.0 <= values[name] < 1.0:
                raise ValueError('{} must be in [0, 1)'.format(name))
        for name in ('tau_down', 'tau_up', 'tau_rec', 'tau_accept',
                     'kappa_g'):
            if values[name] < 0.0:
                raise ValueError('{} must be non-negative'.format(name))
        for name in ('p_good', 'p_bad', 't_rec', 't_trial', 'grad_window'):
            if isinstance(values[name], bool) or int(values[name]) != values[name] or values[name] < 1:
                raise ValueError('{} must be a positive integer'.format(name))
        for name in ('gamma_good', 'gamma_down', 'gamma_safe'):
            if not 0.0 < values[name] <= 1.0:
                raise ValueError('{} must be in (0, 1]'.format(name))
        if values['gamma_up'] <= 1.0:
            raise ValueError('gamma_up must be greater than 1')
        if values['epsilon'] <= 0.0:
            raise ValueError('epsilon must be positive')
        if values['eta_min'] <= 0.0:
            raise ValueError('eta_min must be positive')
        if values['eta_max'] < values['eta_min']:
            raise ValueError('eta_max must be greater than or equal to eta_min')

    @staticmethod
    def compute_gradient_norm(parameters):
        """Return the global L2 norm of the currently accumulated gradients."""
        squared_norm = 0.0
        for parameter in parameters:
            if parameter.grad is None:
                continue
            grad_norm = parameter.grad.detach().norm(2).item()
            squared_norm += grad_norm * grad_norm
        return math.sqrt(squared_norm)

    def step(self, validation_loss=None, train_epoch_loss=None,
             epoch_grad_norm=None):
        """Observe one epoch and update the controller.

        Validation loss is selected whenever it is supplied; otherwise the
        epoch-average training loss is used. The result describes the action
        taken at this epoch boundary.
        """
        monitor_loss = validation_loss
        loss_source = 'validation'
        if monitor_loss is None:
            monitor_loss = train_epoch_loss
            loss_source = 'train_epoch_mean'
        if monitor_loss is None:
            raise ValueError('validation_loss or train_epoch_loss is required')
        if epoch_grad_norm is None:
            raise ValueError('epoch_grad_norm is required')

        monitor_loss = self._finite_float(monitor_loss, 'monitor loss')
        epoch_grad_norm = self._finite_float(
            epoch_grad_norm, 'epoch gradient norm')
        if epoch_grad_norm < 0.0:
            raise ValueError('epoch_grad_norm must be non-negative')

        relative_loss_change = self._update_loss_ema(monitor_loss)
        relative_gradient = self._update_gradient_ema(epoch_grad_norm)
        trend = self._classify_trend(relative_loss_change)
        self.steps += 1

        event = 'initialized' if relative_loss_change is None else 'lr_held'
        rolled_back = False
        recovery_improvement = None
        trial_improvement = None

        if self.state == self.NORMAL:
            event = self._step_normal(trend, relative_loss_change)
        elif self.state == self.RECOVERY:
            event, recovery_improvement = self._step_recovery(
                relative_gradient)
        elif self.state == self.TRIAL:
            event, trial_improvement, rolled_back = self._step_trial()
        else:
            raise RuntimeError('invalid controller state: {}'.format(self.state))

        return {
            'state': self.state,
            'event': event,
            'loss_source': loss_source,
            'learning_rate': self._current_lr(),
            'smoothed_loss': self.smoothed_loss,
            'relative_loss_change': relative_loss_change,
            'loss_trend': trend,
            'smoothed_grad_norm': self.smoothed_grad_norm,
            'relative_gradient': relative_gradient,
            'recovery_improvement': recovery_improvement,
            'trial_improvement': trial_improvement,
            'rolled_back': rolled_back,
        }

    def _step_normal(self, trend, relative_loss_change):
        if relative_loss_change is None:
            self._reset_trend_counters()
            return 'initialized'

        if trend == 'down':
            self.good_count += 1
            self.bad_count = 0
            if self.good_count >= self.p_good:
                self._set_lr(self.gamma_good * self._current_lr())
                self.good_count = 0
                return 'good_decay'
            return 'good_observed'

        if trend == 'up':
            self.bad_count += 1
            self.good_count = 0
            if self.bad_count > self.p_bad:
                self.base_loss = self.smoothed_loss
                self.pre_reduction_lr = self._current_lr()
                self.recovery_lr = self._clip_lr(
                    self.gamma_down * self.pre_reduction_lr)
                self._set_lr(self.recovery_lr)
                self._start_recovery_window()
                self._reset_trend_counters()
                self.state = self.RECOVERY
                return 'recovery_started'
            return 'bad_tolerated'

        self._reset_trend_counters()
        return 'stable'

    def _step_recovery(self, relative_gradient):
        self._set_lr(self.recovery_lr)
        self.recovery_count += 1
        if self.recovery_best_loss is None:
            self.recovery_best_loss = self.smoothed_loss
        else:
            self.recovery_best_loss = min(
                self.recovery_best_loss, self.smoothed_loss)

        if self.recovery_count < self.t_rec:
            return 'recovery_observing', None

        improvement = self._relative_improvement(
            self.base_loss, self.recovery_best_loss)
        if improvement >= self.tau_rec:
            self.state = self.NORMAL
            self._reset_windows()
            self._reset_trend_counters()
            return 'recovery_accepted', improvement

        if relative_gradient < self.kappa_g:
            trial_lr = self._clip_lr(min(
                self.gamma_up * self.recovery_lr,
                self.pre_reduction_lr,
                self.eta_max))
            if trial_lr > self.recovery_lr + self.epsilon:
                self.saved_lr = self._current_lr()
                self._save_trial_checkpoint()
                self.state = self.TRIAL
                self.trial_count = 0
                self.trial_best_loss = None
                self._reset_trend_counters()
                self._set_lr(trial_lr)
                return 'trial_started', improvement

            self._restart_recovery_after_decay()
            return 'trial_skipped_at_lr_bound', improvement

        self._restart_recovery_after_decay()
        return 'recovery_reduced_again', improvement

    def _step_trial(self):
        self.trial_count += 1
        if self.trial_best_loss is None:
            self.trial_best_loss = self.smoothed_loss
        else:
            self.trial_best_loss = min(
                self.trial_best_loss, self.smoothed_loss)

        if self.trial_count < self.t_trial:
            return 'trial_observing', None, False

        improvement = self._relative_improvement(
            self.base_loss, self.trial_best_loss)
        if improvement >= self.tau_accept:
            self.state = self.NORMAL
            self._reset_windows()
            self._reset_trend_counters()
            self._discard_trial_checkpoint()
            return 'trial_accepted', improvement, False

        self.state = self.ROLLBACK
        saved_lr = self._restore_trial_checkpoint()
        safe_lr = self._clip_lr(self.gamma_safe * saved_lr)
        self._set_lr(safe_lr)
        self.state = self.NORMAL
        self._reset_windows()
        self._reset_trend_counters()
        self._discard_trial_checkpoint()
        return 'trial_rolled_back', improvement, True

    def _restart_recovery_after_decay(self):
        current_lr = self._current_lr()
        self.base_loss = self.smoothed_loss
        self.pre_reduction_lr = current_lr
        self.recovery_lr = self._clip_lr(self.gamma_down * current_lr)
        self._set_lr(self.recovery_lr)
        self._start_recovery_window()
        self._reset_trend_counters()
        self.state = self.RECOVERY

    def _start_recovery_window(self):
        self.recovery_count = 0
        self.recovery_best_loss = None
        self.trial_count = 0
        self.trial_best_loss = None
        self.saved_lr = None

    def _reset_windows(self):
        self.base_loss = None
        self.pre_reduction_lr = None
        self.recovery_lr = None
        self.recovery_count = 0
        self.recovery_best_loss = None
        self.trial_count = 0
        self.trial_best_loss = None
        self.saved_lr = None

    def _reset_trend_counters(self):
        self.good_count = 0
        self.bad_count = 0

    def _update_loss_ema(self, loss):
        previous = self.smoothed_loss
        if previous is None:
            self.smoothed_loss = loss
            return None
        self.smoothed_loss = (
            self.beta * previous + (1.0 - self.beta) * loss)
        return ((self.smoothed_loss - previous) /
                (abs(previous) + self.epsilon))

    def _update_gradient_ema(self, gradient_norm):
        if self.smoothed_grad_norm is None:
            self.smoothed_grad_norm = gradient_norm
        else:
            self.smoothed_grad_norm = (
                self.beta_g * self.smoothed_grad_norm
                + (1.0 - self.beta_g) * gradient_norm)
        self.gradient_history.append(self.smoothed_grad_norm)
        if len(self.gradient_history) > self.grad_window:
            self.gradient_history = self.gradient_history[-self.grad_window:]
        reference = float(np.median(self.gradient_history))
        return self.smoothed_grad_norm / (reference + self.epsilon)

    def _classify_trend(self, relative_loss_change):
        if relative_loss_change is None:
            return 'uninitialized'
        if relative_loss_change < -self.tau_down:
            return 'down'
        if relative_loss_change > self.tau_up:
            return 'up'
        return 'stable'

    def _relative_improvement(self, baseline, best_loss):
        return ((baseline - best_loss) /
                (abs(baseline) + self.epsilon))

    def _current_lr(self):
        return float(self.optimizer.param_groups[0]['lr'])

    def _clip_lr(self, learning_rate):
        return min(max(float(learning_rate), self.eta_min), self.eta_max)

    def _set_lr(self, learning_rate):
        learning_rate = self._clip_lr(learning_rate)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

    @staticmethod
    def _finite_float(value, name):
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError('{} must be scalar'.format(name))
            value = value.detach().item()
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('{} must be finite'.format(name))
        return value

    def _save_trial_checkpoint(self):
        parent = os.path.dirname(os.path.abspath(self.checkpoint_path))
        os.makedirs(parent, exist_ok=True)
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'controller': self.state_dict(),
            'scaler': (self.scaler.state_dict()
                       if self.scaler is not None else None),
            'learning_rate': self._current_lr(),
            'reference_loss': self.base_loss,
            'rng': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch_cpu': torch.get_rng_state(),
                'torch_cuda': (torch.cuda.get_rng_state_all()
                               if torch.cuda.is_available() else None),
            },
        }
        torch.save(checkpoint, self.checkpoint_path)

    def _restore_trial_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            raise RuntimeError(
                'trial checkpoint does not exist: {}'.format(
                    self.checkpoint_path))
        device = next(self.model.parameters(), torch.empty(0)).device
        checkpoint = torch.load(
            self.checkpoint_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scaler is not None and checkpoint['scaler'] is not None:
            self.scaler.load_state_dict(checkpoint['scaler'])
        self.load_state_dict(checkpoint['controller'])

        rng = checkpoint['rng']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch_cpu'].cpu())
        if torch.cuda.is_available() and rng['torch_cuda'] is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in rng['torch_cuda']])
        return float(checkpoint['learning_rate'])

    def _discard_trial_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)

    def state_dict(self):
        """Return serializable controller state without checkpoint contents."""
        return {
            'version': self._STATE_DICT_VERSION,
            'hyperparameters': {
                'beta': self.beta,
                'beta_g': self.beta_g,
                'tau_down': self.tau_down,
                'tau_up': self.tau_up,
                'p_good': self.p_good,
                'p_bad': self.p_bad,
                't_rec': self.t_rec,
                't_trial': self.t_trial,
                'gamma_good': self.gamma_good,
                'gamma_down': self.gamma_down,
                'gamma_up': self.gamma_up,
                'gamma_safe': self.gamma_safe,
                'tau_rec': self.tau_rec,
                'tau_accept': self.tau_accept,
                'grad_window': self.grad_window,
                'kappa_g': self.kappa_g,
                'epsilon': self.epsilon,
                'eta_min': self.eta_min,
                'eta_max': self.eta_max,
            },
            'state': self.state,
            'steps': self.steps,
            'smoothed_loss': self.smoothed_loss,
            'smoothed_grad_norm': self.smoothed_grad_norm,
            'gradient_history': list(self.gradient_history),
            'good_count': self.good_count,
            'bad_count': self.bad_count,
            'base_loss': self.base_loss,
            'pre_reduction_lr': self.pre_reduction_lr,
            'recovery_lr': self.recovery_lr,
            'recovery_count': self.recovery_count,
            'recovery_best_loss': self.recovery_best_loss,
            'trial_count': self.trial_count,
            'trial_best_loss': self.trial_best_loss,
            'saved_lr': self.saved_lr,
        }

    def load_state_dict(self, state_dict):
        """Restore controller state; model and optimizer are restored separately."""
        if state_dict.get('version') != self._STATE_DICT_VERSION:
            raise ValueError('unsupported controller state version')
        hyperparameters = state_dict['hyperparameters']
        for name, value in hyperparameters.items():
            setattr(self, name, value)

        self.state = state_dict['state']
        self.steps = state_dict['steps']
        self.smoothed_loss = state_dict['smoothed_loss']
        self.smoothed_grad_norm = state_dict['smoothed_grad_norm']
        self.gradient_history = list(state_dict['gradient_history'])
        self.good_count = state_dict['good_count']
        self.bad_count = state_dict['bad_count']
        self.base_loss = state_dict['base_loss']
        self.pre_reduction_lr = state_dict['pre_reduction_lr']
        self.recovery_lr = state_dict['recovery_lr']
        self.recovery_count = state_dict['recovery_count']
        self.recovery_best_loss = state_dict['recovery_best_loss']
        self.trial_count = state_dict['trial_count']
        self.trial_best_loss = state_dict['trial_best_loss']
        self.saved_lr = state_dict['saved_lr']

    def close(self):
        """Remove an unresolved trial checkpoint when training terminates."""
        self._discard_trial_checkpoint()


class PlateauLRController:
    """Minimal loss-plateau learning-rate controller.

    This first-stage ablation deliberately uses only the historical best
    monitored loss. It does not use EMA, gradients, exploration, or rollback.
    """

    requires_gradient = False

    def __init__(self, optimizer, patience=3, factor=0.9, eta_min=1e-7):
        if not optimizer.param_groups:
            raise ValueError('optimizer must contain at least one parameter group')
        if isinstance(patience, bool) or int(patience) != patience or patience < 1:
            raise ValueError('patience must be a positive integer')
        if not 0.0 < factor < 1.0:
            raise ValueError('factor must be in (0, 1)')
        if eta_min <= 0.0:
            raise ValueError('eta_min must be positive')

        self.optimizer = optimizer
        self.patience = int(patience)
        self.factor = float(factor)
        self.eta_min = float(eta_min)
        self.state = 'plateau'
        self.best_loss = None
        self.bad_epochs = 0
        self.reductions = 0

        self._set_lr(self._current_lr())

    def step(self, validation_loss=None, train_epoch_loss=None):
        """Observe one epoch and hold or reduce the learning rate.

        Validation loss has priority. If it is unavailable, train_epoch_loss
        must be the mean over the complete epoch, not the final batch loss.
        """
        monitor_loss = validation_loss
        loss_source = 'validation'
        if monitor_loss is None:
            monitor_loss = train_epoch_loss
            loss_source = 'train_epoch_mean'
        if monitor_loss is None:
            raise ValueError('validation_loss or train_epoch_loss is required')

        monitor_loss = self._finite_float(monitor_loss, 'monitor loss')
        improved = self.best_loss is None or monitor_loss < self.best_loss
        if improved:
            self.best_loss = monitor_loss
            self.bad_epochs = 0
            event = 'improved'
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                old_lr = self._current_lr()
                new_lr = self._clip_lr(self.factor * old_lr)
                self._set_lr(new_lr)
                self.bad_epochs = 0
                if new_lr < old_lr:
                    self.reductions += 1
                    event = 'lr_reduced'
                else:
                    event = 'lr_floor'
            else:
                event = 'lr_held'

        return {
            'state': self.state,
            'event': event,
            'loss_source': loss_source,
            'learning_rate': self._current_lr(),
            'best_loss': self.best_loss,
            'bad_epochs': self.bad_epochs,
            'patience': self.patience,
            'reductions': self.reductions,
        }

    def state_dict(self):
        return {
            'state': self.state,
            'patience': self.patience,
            'factor': self.factor,
            'eta_min': self.eta_min,
            'best_loss': self.best_loss,
            'bad_epochs': self.bad_epochs,
            'reductions': self.reductions,
        }

    def load_state_dict(self, state_dict):
        self.state = state_dict['state']
        self.patience = int(state_dict['patience'])
        self.factor = float(state_dict['factor'])
        self.eta_min = float(state_dict['eta_min'])
        self.best_loss = state_dict['best_loss']
        self.bad_epochs = int(state_dict['bad_epochs'])
        self.reductions = int(state_dict['reductions'])

    def close(self):
        pass

    def _current_lr(self):
        return float(self.optimizer.param_groups[0]['lr'])

    def _clip_lr(self, learning_rate):
        return max(float(learning_rate), self.eta_min)

    def _set_lr(self, learning_rate):
        learning_rate = self._clip_lr(learning_rate)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

    @staticmethod
    def _finite_float(value, name):
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError('{} must be scalar'.format(name))
            value = value.detach().item()
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('{} must be finite'.format(name))
        return value


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
