import unittest
from typing import Dict, Union, Any

import torch
from torch import nn
from transformers import Trainer

class LDMTrainer(Trainer):

    # def create_optimizer(self):
    #     optimizer = Ada(self.model.parameters(), lr=2e-4)
    #     return optimizer

    def training_step(self, model, inputs: Dict[str, Union[torch.Tensor, Any]]):
        loss = model.compute_loss(x0=inputs['input'], c=inputs['cond'], uc=inputs['uncond'])
        return loss



# class MyTestCase(unittest.TestCase):
#     def test_something(self):
#         self.assertEqual(True, False)  # add assertion here
#

if __name__ == '__main__':
    unittest.main()
