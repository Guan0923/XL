import copy
import os
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn, optim

from utils.tools import LossGradientFeedbackLRController


class DummyScaler:
    def __init__(self, scale=64.0):
        self.scale = scale

    def state_dict(self):
        return {'scale': self.scale}

    def load_state_dict(self, state_dict):
        self.scale = state_dict['scale']


def assert_nested_equal(test_case, expected, actual):
    if torch.is_tensor(expected):
        test_case.assertTrue(torch.equal(expected, actual))
    elif isinstance(expected, dict):
        test_case.assertEqual(expected.keys(), actual.keys())
        for key in expected:
            assert_nested_equal(test_case, expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        test_case.assertEqual(len(expected), len(actual))
        for expected_item, actual_item in zip(expected, actual):
            assert_nested_equal(test_case, expected_item, actual_item)
    else:
        test_case.assertEqual(expected, actual)


class LossGradientFeedbackLRControllerTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=os.path.dirname(__file__))

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_controller(self, scaler=None, **overrides):
        model = nn.Linear(1, 1)
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        options = {
            'beta': 0.0,
            'beta_g': 0.0,
            'tau_down': 0.001,
            'tau_up': 0.001,
            'eta_min': 0.001,
            'eta_max': 0.1,
        }
        options.update(overrides)
        controller = LossGradientFeedbackLRController(
            model,
            optimizer,
            os.path.join(self.temp_dir.name, 'trial.pth'),
            scaler=scaler,
            **options)
        return model, optimizer, controller

    @staticmethod
    def start_recovery(controller, gradient_norm=1.0):
        controller.step(1.00, None, gradient_norm)
        first = controller.step(1.02, None, gradient_norm)
        second = controller.step(1.04, None, gradient_norm)
        third = controller.step(1.06, None, gradient_norm)
        return first, second, third

    def test_validation_loss_has_priority_and_gradient_norm_is_global_l2(self):
        model, _, controller = self.make_controller()
        result = controller.step(
            validation_loss=1.0,
            train_epoch_loss=9.0,
            epoch_grad_norm=1.0)
        self.assertEqual(result['loss_source'], 'validation')
        self.assertEqual(result['smoothed_loss'], 1.0)

        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter) * 3.0
        expected = math_sqrt_sum_of_gradient_squares(model.parameters())
        actual = controller.compute_gradient_norm(model.parameters())
        self.assertAlmostEqual(expected, actual)

    def test_third_rise_starts_recovery_and_resets_counters(self):
        _, optimizer, controller = self.make_controller()
        first, second, third = self.start_recovery(controller)

        self.assertEqual(first['event'], 'bad_tolerated')
        self.assertEqual(second['event'], 'bad_tolerated')
        self.assertEqual(third['event'], 'recovery_started')
        self.assertEqual(controller.state, controller.RECOVERY)
        self.assertEqual(controller.good_count, 0)
        self.assertEqual(controller.bad_count, 0)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.05)

    def test_stable_epoch_breaks_consecutive_trend(self):
        _, _, controller = self.make_controller()
        controller.step(1.0, None, 1.0)
        controller.step(1.1, None, 1.0)
        self.assertEqual(controller.bad_count, 1)
        controller.step(1.1, None, 1.0)
        self.assertEqual(controller.bad_count, 0)
        self.assertEqual(controller.good_count, 0)

    def test_good_decay_respects_minimum_lr(self):
        _, optimizer, controller = self.make_controller(
            gamma_good=0.5, eta_min=0.08)
        controller.step(1.0, None, 1.0)
        controller.step(0.9, None, 1.0)
        result = controller.step(0.8, None, 1.0)
        self.assertEqual(result['event'], 'good_decay')
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.08)

    def test_recovery_uses_exact_window_and_accepts_improvement(self):
        _, _, controller = self.make_controller()
        self.start_recovery(controller)

        first = controller.step(1.04, None, 1.0)
        second = controller.step(1.02, None, 1.0)
        third = controller.step(0.98, None, 1.0)

        self.assertEqual(first['event'], 'recovery_observing')
        self.assertEqual(second['event'], 'recovery_observing')
        self.assertEqual(third['event'], 'recovery_accepted')
        self.assertEqual(controller.state, controller.NORMAL)

    def test_non_small_gradient_reduces_again_and_restarts_recovery(self):
        _, optimizer, controller = self.make_controller()
        self.start_recovery(controller)
        controller.step(1.07, None, 1.0)
        controller.step(1.08, None, 1.0)
        result = controller.step(1.09, None, 1.0)

        self.assertEqual(result['event'], 'recovery_reduced_again')
        self.assertEqual(controller.state, controller.RECOVERY)
        self.assertEqual(controller.recovery_count, 0)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.025)

    def test_small_relative_gradient_starts_bounded_trial(self):
        _, optimizer, controller = self.make_controller()
        self.start_recovery(controller)
        controller.step(1.07, None, 1.0)
        controller.step(1.08, None, 1.0)
        result = controller.step(1.09, None, 0.01)

        self.assertEqual(result['event'], 'trial_started')
        self.assertEqual(controller.state, controller.TRIAL)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.0625)
        self.assertLessEqual(optimizer.param_groups[0]['lr'], 0.1)
        self.assertTrue(os.path.exists(controller.checkpoint_path))

    def test_trial_uses_exact_window_and_accepts_improvement(self):
        _, _, controller = self.make_controller()
        self.start_recovery(controller)
        controller.step(1.07, None, 1.0)
        controller.step(1.08, None, 1.0)
        controller.step(1.09, None, 0.01)

        first = controller.step(1.00, None, 1.0)
        second = controller.step(0.99, None, 1.0)

        self.assertEqual(first['event'], 'trial_observing')
        self.assertEqual(first['state'], controller.TRIAL)
        self.assertEqual(second['event'], 'trial_accepted')
        self.assertEqual(controller.state, controller.NORMAL)
        self.assertFalse(os.path.exists(controller.checkpoint_path))

    def test_failed_trial_restores_full_training_state_and_rng(self):
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        scaler = DummyScaler()
        model, optimizer, controller = self.make_controller(scaler=scaler)
        train_one_step(model, optimizer)
        self.start_recovery(controller)
        controller.step(1.07, None, 1.0)
        controller.step(1.08, None, 1.0)

        saved_model = copy.deepcopy(model.state_dict())
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        saved_scaler = copy.deepcopy(scaler.state_dict())
        started = controller.step(1.09, None, 0.01)
        saved_controller = torch.load(
            controller.checkpoint_path, weights_only=False)['controller']
        self.assertEqual(started['event'], 'trial_started')

        expected_python = random.random()
        expected_numpy = np.random.rand()
        expected_torch = torch.rand(1)

        train_one_step(model, optimizer)
        scaler.scale = 8.0
        random.random()
        np.random.rand()
        torch.rand(1)
        controller.step(1.10, None, 1.0)
        train_one_step(model, optimizer)
        result = controller.step(1.20, None, 1.0)

        self.assertEqual(result['event'], 'trial_rolled_back')
        self.assertTrue(result['rolled_back'])
        self.assertEqual(controller.state, controller.NORMAL)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.025)
        assert_nested_equal(self, saved_model, model.state_dict())
        assert_nested_equal(self, saved_optimizer['state'],
                            optimizer.state_dict()['state'])
        self.assertEqual(saved_scaler, scaler.state_dict())
        self.assertEqual(controller.smoothed_loss,
                         saved_controller['smoothed_loss'])
        self.assertEqual(controller.gradient_history,
                         saved_controller['gradient_history'])
        self.assertEqual(expected_python, random.random())
        self.assertEqual(expected_numpy, np.random.rand())
        self.assertTrue(torch.equal(expected_torch, torch.rand(1)))
        self.assertFalse(os.path.exists(controller.checkpoint_path))

    def test_state_dict_round_trip(self):
        _, _, controller = self.make_controller()
        controller.step(None, 1.0, 2.0)
        controller.step(None, 1.1, 3.0)
        saved = copy.deepcopy(controller.state_dict())

        _, _, restored = self.make_controller(beta=0.5)
        restored.load_state_dict(saved)
        self.assertEqual(saved, restored.state_dict())


def math_sqrt_sum_of_gradient_squares(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float((parameter.grad ** 2).sum())
    return total ** 0.5


def train_one_step(model, optimizer):
    optimizer.zero_grad()
    prediction = model(torch.tensor([[1.0]]))
    loss = (prediction - torch.tensor([[0.0]])).pow(2).mean()
    loss.backward()
    optimizer.step()


if __name__ == '__main__':
    unittest.main()
