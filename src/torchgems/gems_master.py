# Copyright 2023, The Ohio State University. All rights reserved.
# The Infer-HiRes software package is developed by the team members of
# The Ohio State University's Network-Based Computing Laboratory (NBCL),
# headed by Professor Dhabaleswar K. (DK) Panda.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torchgems.mp_pipeline import train_model
import torch


class train_model_master:
    def __init__(
        self,
        model_gen1,
        model_gen2,
        local_rank,
        batch_size,
        epochs,
        precision,
        eval_mode,
        criterion=None,
        optimizer=None,
        parts=1,
        ASYNC=True,
        replications=1,
    ):
        self.mp_size = model_gen1.split_size
        self.split_size = model_gen1.split_size
        self.second_rank = self.split_size - local_rank - 1

        self.train_model1 = train_model(
            model_gen1,
            local_rank,
            batch_size,
            epochs,
            precision,
            eval_mode=eval_mode,
            criterion=None,
            optimizer=None,
            parts=parts,
            ASYNC=True,
            GEMS_INVERSE=False,
        )
        self.train_model2 = train_model(
            model_gen2,
            self.second_rank,
            batch_size,
            epochs,
            precision,
            eval_mode=eval_mode,
            criterion=None,
            optimizer=None,
            parts=parts,
            ASYNC=True,
            GEMS_INVERSE=True,
        )

        self.parts = parts
        self.epochs = epochs
        self.local_rank = local_rank
        self.ENABLE_ASYNC = ASYNC
        self.batch_size = batch_size

        self.replications = replications

    def run_step(self, inputs, labels, eval_mode):
        loss, correct = 0, 0
        temp_loss, temp_correct = self.train_model1.run_step(
            inputs[: self.batch_size], labels[: self.batch_size], eval_mode
        )
        loss += temp_loss
        correct += temp_correct
        temp_loss, temp_correct = self.train_model2.run_step(
            inputs[self.batch_size : 2 * self.batch_size],
            labels[self.batch_size : 2 * self.batch_size],
            eval_mode,
        )
        loss += temp_loss
        correct += temp_correct

        torch.cuda.synchronize()
        for times in range(self.replications - 1):
            index = (2 * times) + 2
            temp_loss, temp_correct = self.train_model1.run_step(
                inputs[index * self.batch_size : (index + 1) * self.batch_size],
                labels[index * self.batch_size : (index + 1) * self.batch_size],
                eval_mode,
            )
            loss += temp_loss
            correct += temp_correct

            temp_loss, temp_correct = self.train_model2.run_step(
                inputs[(index + 1) * self.batch_size : (index + 2) * self.batch_size],
                labels[(index + 1) * self.batch_size : (index + 2) * self.batch_size],
                eval_mode,
            )

            loss += temp_loss
            correct += temp_correct
        return loss, correct
