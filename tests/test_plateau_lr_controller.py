import unittest

from torch import nn, optim

from utils.tools import PlateauLRController


class PlateauLRControllerTest(unittest.TestCase):
    def make_controller(self, eta_min=1e-7):
        model = nn.Linear(1, 1)
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        controller = PlateauLRController(
            optimizer=optimizer,
            patience=3,
            factor=0.5,
            eta_min=eta_min)
        return optimizer, controller

    def test_improvement_holds_lr_and_three_bad_epochs_halve(self):
        optimizer, controller = self.make_controller()

        first = controller.step(validation_loss=1.0, train_epoch_loss=9.0)
        second = controller.step(validation_loss=1.1, train_epoch_loss=9.0)
        third = controller.step(validation_loss=1.2, train_epoch_loss=9.0)
        fourth = controller.step(validation_loss=1.3, train_epoch_loss=9.0)

        self.assertEqual(first['event'], 'improved')
        self.assertEqual(second['event'], 'lr_held')
        self.assertEqual(third['event'], 'lr_held')
        self.assertEqual(fourth['event'], 'lr_halved')
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.05)
        self.assertEqual(controller.bad_epochs, 0)

        improved = controller.step(validation_loss=0.9, train_epoch_loss=9.0)
        self.assertEqual(improved['event'], 'improved')
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.05)
        self.assertEqual(controller.bad_epochs, 0)

    def test_counter_resets_and_halves_again_after_three_bad_epochs(self):
        optimizer, controller = self.make_controller()

        for loss in (1.0, 1.1, 1.2, 1.3):
            result = controller.step(validation_loss=loss)
        self.assertEqual(result['event'], 'lr_halved')

        held = controller.step(validation_loss=1.4)
        self.assertEqual(held['event'], 'lr_held')
        self.assertEqual(held['bad_epochs'], 1)
        controller.step(validation_loss=1.5)
        second_halving = controller.step(validation_loss=1.6)

        self.assertEqual(second_halving['event'], 'lr_halved')
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.025)
        self.assertEqual(controller.reductions, 2)

    def test_eta_min_and_metric_fallback(self):
        optimizer, controller = self.make_controller(eta_min=0.04)

        controller.step(validation_loss=None, train_epoch_loss=1.0)
        controller.step(validation_loss=None, train_epoch_loss=1.1)
        controller.step(validation_loss=None, train_epoch_loss=1.2)
        first = controller.step(validation_loss=None, train_epoch_loss=1.3)
        self.assertEqual(first['loss_source'], 'train_epoch_mean')
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.05)

        controller.step(validation_loss=1.4, train_epoch_loss=0.1)
        controller.step(validation_loss=1.5, train_epoch_loss=0.1)
        floor = controller.step(validation_loss=1.6, train_epoch_loss=0.1)
        self.assertEqual(floor['event'], 'lr_halved')
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.04)
        self.assertGreaterEqual(optimizer.param_groups[0]['lr'], 0.04)
        self.assertFalse(controller.requires_gradient)

    def test_validation_loss_has_priority(self):
        _, controller = self.make_controller()
        result = controller.step(validation_loss=1.0, train_epoch_loss=0.1)
        self.assertEqual(result['loss_source'], 'validation')
        self.assertEqual(result['best_loss'], 1.0)


if __name__ == '__main__':
    unittest.main()