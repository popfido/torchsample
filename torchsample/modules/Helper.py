# coding=utf-8
from __future__ import print_function
from __future__ import absolute_import

import functools
import torch as th


class BaseHelper(object):

    def move_to_cpu(self, outputs):
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            if len(outputs) == 0:
                raise ValueError("There is nothing in outputs tuple")
            for o in outputs:
                if not isinstance(o, th.Tensor):
                    raise TypeError("There exists non torch.Tensor Entity")
            if isinstance(outputs, tuple):
                return tuple(map(lambda output: output.cpu(), outputs))
            else:
                return [output.cpu() for output in outputs]
        else:
            if isinstance(outputs, th.Tensor):
                return outputs.cpu()
            else:
                raise TypeError("There exists non torch.Tensor Entity")


class SingleInputSingleTargetHelper(BaseHelper):

    def move_to_cuda(self, cuda_device, inputs, targets):
        inputs = inputs.cuda(cuda_device)
        targets = targets.cuda(cuda_device)
        return inputs, targets

    def shuffle_arrays(self, inputs, targets):
        rand_indices = th.randperm(len(inputs))
        inputs = inputs[rand_indices]
        targets = targets[rand_indices]
        return inputs, targets

    def grab_batch(self, batch_idx, batch_size, inputs, targets):
        input_batch = th.tensor(inputs[batch_idx * batch_size:(batch_idx + 1) * batch_size], requires_grad=True)
        target_batch = th.tensor(targets[batch_idx * batch_size:(batch_idx + 1) * batch_size])
        return input_batch, target_batch

    def grab_batch_from_loader(self, loader_iter):
        input_batch, target_batch = next(loader_iter)
        return th.tensor(input_batch, requires_grad=True), th.tensor(target_batch)

    def apply_transforms(self, tforms, input_batch, target_batch):
        input_batch = tforms[0](input_batch)
        target_batch = tforms[1](target_batch)
        input_batch, target_batch = tforms[2](input_batch, target_batch)
        return input_batch, target_batch

    def forward_pass(self, input_batch, model):
        return model(input_batch)

    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)

    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return loss_fn(output_batch, target_batch)

    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)
        # def new_loss_fn(output_batch, target_batch):
        #    return self.calculate_loss(output_batch, target_batch, loss_fn)
        # return new_loss_fn


class SingleInputMultiTargetHelper(BaseHelper):

    def move_to_cuda(self, cuda_device, inputs, targets):
        inputs = inputs.cuda(cuda_device)
        targets = [target_.cuda(cuda_device) for target_ in targets]
        return inputs, targets

    def shuffle_arrays(self, inputs, targets):
        rand_indices = th.randperm(len(inputs))
        inputs = inputs[rand_indices]
        targets = [target_[rand_indices] for target_ in targets]
        return inputs, targets

    def grab_batch(self, batch_idx, batch_size, inputs, targets):
        input_batch = th.tensor(inputs[batch_idx * batch_size:(batch_idx + 1) * batch_size], requires_grad=True)
        target_batch = [th.tensor(target_[batch_idx * batch_size:(batch_idx + 1) * batch_size])
                        for target_ in targets]
        return input_batch, target_batch

    def grab_batch_from_loader(self, loader_iter):
        input_batch, target_batch = next(loader_iter)
        return th.tensor(input_batch, requires_grad=True), [th.tensor(target_) for target_ in target_batch]

    def apply_transforms(self, tforms, input_batch, target_batch):
        input_batch = tforms[0](input_batch)
        target_batch = [tforms[1](target_) for target_ in target_batch]
        return input_batch, target_batch

    def forward_pass(self, input_batch, model):
        return model(input_batch)

    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)

    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return sum([loss_fn[idx](output_batch[idx], target_batch[idx])
                    for idx in range(len(output_batch))])

    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)


class MultiInputSingleTargetHelper(BaseHelper):

    def move_to_cuda(self, cuda_device, inputs, targets):
        inputs = [input_.cuda(cuda_device) for input_ in inputs]
        targets = targets.cuda(cuda_device)
        return inputs, targets

    def shuffle_arrays(self, inputs, targets):
        rand_indices = th.randperm(len(inputs))
        inputs = [input_[rand_indices] for input_ in inputs]
        targets = targets[rand_indices]
        return inputs, targets

    def grab_batch(self, batch_idx, batch_size, inputs, targets):
        input_batch = [th.tensor(input_[batch_idx * batch_size:(batch_idx + 1) * batch_size], requires_grad=True)
                       for input_ in inputs]
        target_batch = th.tensor(targets[batch_idx * batch_size:(batch_idx + 1) * batch_size])
        return input_batch, target_batch

    def grab_batch_from_loader(self, loader_iter):
        input_batch, target_batch = next(loader_iter)
        return [th.tensor(input_, requires_grad=True) for input_ in input_batch], th.tensor(target_batch)

    def apply_transforms(self, tforms, input_batch, target_batch):
        input_batch = [tforms[0](input_) for input_ in input_batch]
        target_batch = tforms[1](target_batch)
        return input_batch, target_batch

    def forward_pass(self, input_batch, model):
        return model(*input_batch)

    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)

    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return loss_fn(output_batch, target_batch)

    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)


class MultiInputMultiTargetHelper(BaseHelper):

    def move_to_cuda(self, cuda_device, inputs, targets):
        inputs = [input_.cuda(cuda_device) for input_ in inputs]
        targets = [target_.cuda(cuda_device) for target_ in targets]
        return inputs, targets

    def shuffle_arrays(self, inputs, targets):
        rand_indices = th.randperm(len(inputs))
        inputs = [input_[rand_indices] for input_ in inputs]
        targets = [input_[rand_indices] for input_ in inputs]
        return inputs, targets

    def grab_batch(self, batch_idx, batch_size, inputs, targets, volatile=False):
        input_batch = [th.tensor(input_[batch_idx * batch_size:(batch_idx + 1) * batch_size], requires_grads=True)
                       for input_ in inputs]
        target_batch = [th.tensor(target_[batch_idx * batch_size:(batch_idx + 1) * batch_size])
                        for target_ in targets]
        return input_batch, target_batch

    def grab_batch_from_loader(self, loader_iter, volatile=False):
        input_batch, target_batch = next(loader_iter)
        return [th.tensor(input_, requires_grad=True) for input_ in input_batch], [
            th.tensor(target_) for target_ in target_batch]

    def apply_transforms(self, tforms, input_batch, target_batch):
        input_batch = [tforms[0](input_) for input_ in input_batch]
        target_batch = [tforms[1](target_) for target_ in target_batch]
        return input_batch, target_batch

    def forward_pass(self, input_batch, model):
        return model(*input_batch)

    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)

    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return sum([loss_fn[idx](output_batch[idx], target_batch[idx])
                    for idx in range(len(output_batch))])

    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)


class SingleInputNoTargetHelper(BaseHelper):

    def move_to_cuda(self, cuda_device, inputs, targets=None):
        inputs = inputs.cuda(cuda_device)
        return inputs, None

    def shuffle_arrays(self, inputs, targets=None):
        rand_indices = th.randperm(len(inputs))
        inputs = inputs[rand_indices]
        return inputs, None

    def grab_batch(self, batch_idx, batch_size, inputs, targets=None):
        input_batch = th.tensor(inputs[batch_idx * batch_size:(batch_idx + 1) * batch_size], requires_grad=True)
        return input_batch, None

    def grab_batch_from_loader(self, loader_iter):
        input_batch = next(loader_iter)
        return th.tensor(input_batch, requires_grad=True), None

    def apply_transforms(self, tforms, input_batch):
        input_batch = tforms[0](input_batch)
        return input_batch, None

    def forward_pass(self, input_batch, model):
        return model(input_batch)

    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)

    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return loss_fn(output_batch)

    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)


class MultiInputNoTargetHelper(BaseHelper):

    def move_to_cuda(self, cuda_device, inputs, targets=None):
        inputs = [input_.cuda(cuda_device) for input_ in inputs]
        return inputs, None

    def shuffle_arrays(self, inputs, targets=None):
        rand_indices = th.randperm(len(inputs))
        inputs = [input_[rand_indices] for input_ in inputs]
        return inputs, None

    def grab_batch(self, batch_idx, batch_size, inputs, targets=None):
        input_batch = [th.tensor(input_[batch_idx * batch_size:(batch_idx + 1) * batch_size], requires_grad=True)
                       for input_ in inputs]
        return input_batch, None

    def grab_batch_from_loader(self, loader_iter):
        input_batch = next(loader_iter)
        return [th.tensor(input_, require_grad=True) for input_ in input_batch], None

    def apply_transforms(self, tforms, input_batch, target_batch=None):
        input_batch = [tforms[0](input_) for input_ in input_batch]
        return input_batch, None

    def forward_pass(self, input_batch, model):
        return model(*input_batch)

    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)

    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return loss_fn(output_batch)

    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)