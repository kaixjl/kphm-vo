from typing import List, Optional
import torch.nn as nn
import torch


class SeqAsBatch(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model # type: nn.Module


    def forward(self, input_tensor):
        # type: (torch.Tensor,) -> torch.Tensor
        
        d_input = input_tensor.shape

        input_tensor = input_tensor.reshape((-1, )+d_input[2:])
        output_tensor = self.model(input_tensor) # type: torch.Tensor

        d_output = output_tensor.shape

        output_tensor = output_tensor.reshape(d_input[:2]+d_output[1:])
        return output_tensor

class BatchAsSeq(nn.Module):

    def __init__(self, model, batch_size = None, seq_len = None, batch_first = False):
        super().__init__()
        self.model = model # type: nn.Module
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.batch_first = batch_first
        assert(batch_size is None and seq_len is not None or batch_size is not None and seq_len is None)


    def forward(self, input_tensor):
        # type: (torch.Tensor,) -> torch.Tensor
        
        d_input = input_tensor.shape

        if self.batch_size is not None:
            if self.batch_first:
                input_tensor = input_tensor.reshape((self.batch_size, -1) + d_input[1:])
            else:
                input_tensor = input_tensor.reshape((-1, self.batch_size) + d_input[1:])
        else:
            if self.batch_first:
                input_tensor = input_tensor.reshape((-1, self.seq_len) + d_input[1:])
            else:
                input_tensor = input_tensor.reshape((self.seq_len, -1) + d_input[1:])
        output_tensor = self.model(input_tensor) # type: torch.Tensor

        d_output = output_tensor.shape

        output_tensor = output_tensor.reshape((-1,)+d_output[1:])
        return output_tensor

class FunctorModule(nn.Module):

    def __init__(self, func):
        super().__init__()
        self.func = func


    def forward(self, input_tensor):
        # type: (torch.Tensor,) -> torch.Tensor
        return self.func(input_tensor)

class SeqToBatch(FunctorModule):

    def __init__(self):
        super().__init__(func=lambda x: x.reshape((-1, )+x.shape[2:]))


    # def forward(self, input_tensor):
    #     # type: (torch.Tensor,) -> torch.Tensor
        
    #     d_input = input_tensor.shape

    #     input_tensor = input_tensor.reshape((-1, )+d_input[2:])
    #     return input_tensor

class BatchToSeq(FunctorModule):

    def __init__(self, batch_size = None, seq_len = None, batch_first = False):
        assert(batch_size is None and seq_len is not None or batch_size is not None and seq_len is None)
        if batch_size is not None:
            if batch_first:
                super().__init__(func=lambda x: x.reshape((batch_size, -1) + x.shape[1:]))
            else:
                super().__init__(func=lambda x: x.reshape((-1, batch_size) + x.shape[1:]))
        else:
            if batch_first:
                super().__init__(func=lambda x: x.reshape((-1, seq_len) + x.shape[1:]))
            else:
                super().__init__(func=lambda x: x.reshape((seq_len, -1) + x.shape[1:]))


