import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from exp.exp_main import Exp_Main
from utils.tools import LossGradientFeedbackLRController


class TinyForecastModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 1)

    def forward(self, batch_x, batch_x_mark):
        return self.projection(batch_x)


class TrackingController(LossGradientFeedbackLRController):
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_calls = []
        self.__class__.instances.append(self)

    def step(self, *args, **kwargs):
        self.step_calls.append((args, kwargs))
        return super().step(*args, **kwargs)


class LGFLRExperimentIntegrationTest(unittest.TestCase):
    def test_switch_routes_epoch_metrics_to_controller(self):
        TrackingController.instances.clear()
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as temp_dir:
            experiment = object.__new__(Exp_Main)
            experiment.args = make_args(temp_dir)
            experiment.model = TinyForecastModel()
            experiment.device = torch.device('cpu')
            loader = make_loader()
            experiment._get_data = lambda flag: (None, loader)
            experiment._save_fft_heatmaps = lambda setting, epoch: None

            with mock.patch(
                    'exp.exp_main.LossGradientFeedbackLRController',
                    TrackingController), mock.patch(
                        'exp.exp_main.adjust_learning_rate') as old_scheduler:
                experiment.train('lgflr_smoke')

            self.assertEqual(len(TrackingController.instances), 1)
            controller = TrackingController.instances[0]
            self.assertEqual(len(controller.step_calls), 1)
            _, keyword_args = controller.step_calls[0]
            self.assertIsNotNone(keyword_args['validation_loss'])
            self.assertIsNotNone(keyword_args['train_epoch_loss'])
            self.assertGreater(keyword_args['epoch_grad_norm'], 0.0)
            old_scheduler.assert_not_called()


def make_loader():
    batch_x = torch.tensor([[[1.0], [2.0]], [[2.0], [3.0]]])
    batch_y = torch.tensor([[[0.5], [1.0]], [[1.0], [1.5]]])
    marks = torch.zeros_like(batch_x)
    return [(batch_x, batch_y, marks, marks)]


def make_args(checkpoint_dir):
    return SimpleNamespace(
        checkpoints=checkpoint_dir,
        patience=10,
        learning_rate=0.1,
        use_amp=False,
        use_lgflr=1,
        lradj='type3',
        pct_start=0.3,
        train_epochs=1,
        features='M',
        pred_len=1,
        label_len=1,
        model='XLinear',
        lgf_beta=0.9,
        lgf_beta_g=0.9,
        lgf_tau_down=0.002,
        lgf_tau_up=0.005,
        lgf_p_good=2,
        lgf_p_bad=2,
        lgf_t_rec=3,
        lgf_t_trial=2,
        lgf_gamma_good=0.98,
        lgf_gamma_down=0.5,
        lgf_gamma_up=1.25,
        lgf_gamma_safe=0.5,
        lgf_tau_rec=0.002,
        lgf_tau_accept=0.002,
        lgf_grad_window=5,
        lgf_kappa_g=0.5,
        lgf_epsilon=1e-12,
        lgf_eta_min=1e-7,
        lgf_eta_max=None)


if __name__ == '__main__':
    unittest.main()
