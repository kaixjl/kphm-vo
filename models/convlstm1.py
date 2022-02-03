from typing import List, Optional
import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, use_saved = False):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        use_saved: bool
            If True, save hidden status and cell. User don't need to input cur_state in forward method, and cur_state in forward method is ignored.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self._hidden = None
        self._cell = None
        self.use_saved = use_saved

    def forward(self, input_tensor, cur_state=None):
        # type: (torch.Tensor, List[torch.Tensor, torch.Tensor]) -> torch.Tensor
        '''
        ## Parameters

        - cur_state: tuple(hidden, cell) or list[hidden, cell]. If use_saved = True, saved last time hidden state and cell are used. If use_saved = True and cur_state is provided, cur_state is used. If use_saved = False and cur_state is None, zero state is used.
        '''
        if cur_state is not None:
            h_cur, c_cur = cur_state
        else:
            if self.use_saved:
                if self._hidden is None or self._cell is None:
                    self.init_hidden(input_tensor.shape[0], (input_tensor.shape[2], input_tensor.shape[3]))
                h_cur, c_cur = self._hidden, self._cell
            else:
                cur_state = self.init_hidden(input_tensor.shape[0], (input_tensor.shape[2], input_tensor.shape[3]))
                h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
    # @property
    def hidden(self):
        return self._hidden

    # @property
    def cell(self):
        return self._cell

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        self._hidden = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        self._cell = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        return (self._hidden, self._cell)


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_last_state=False, use_saved = False):
        super(ConvLSTM, self).__init__()

        kernel_size = self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_last_state = return_last_state
        self.use_saved = use_saved
        self._saved_hidden_state = None

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        # type: (torch.Tensor, List[torch.Tensor, torch.Tensor]) -> torch.Tensor
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None or List/Tuple[Tensor, Tensor], each Tensor has shape (num_layers, b, c, h, w)

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            assert(len(hidden_state)==2)
            assert(len(hidden_state[0])==self.num_layers)
            assert(len(hidden_state[1])==self.num_layers)
            for i in range(self.num_layers):
                assert(hidden_state[0][i].shape==(b, self.hidden_dim[i], h, w))
                assert(hidden_state[1][i].shape==(b, self.hidden_dim[i], h, w))
        else:
            if self.use_saved:
                if self._saved_hidden_state is None:
                    hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
                else:
                    hidden_state = self._saved_hidden_state
            else:
                hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        h_n_list = []
        c_n_list = []

        seq_len = input_tensor.shape[1]
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[0][layer_idx], hidden_state[1][layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            h_n_list.append(h)
            c_n_list.append(c)

        last_state_list = (h_n_list, c_n_list)
        self._all_layer_output_list = layer_output_list
        self._all_layer_last_state_list = last_state_list
        self._saved_hidden_state = last_state_list

        # if not self.return_all_layers:
        #     layer_output_list = layer_output_list[-1:]
        #     last_state_list = last_state_list[-1:]

        # return layer_output_list, last_state_list
        if not self.batch_first:
            layer_output = layer_output.permute(1, 0, 2, 3, 4) # (b, t, c, h, w) -> (t, b, c, h, w)
            
        if self.return_last_state:
            return layer_output, last_state_list
        else:
            return layer_output
    
    # @property
    def all_layer_output_list(self):
        return self._all_layer_output_list

    # @property
    def all_layer_last_state_list(self):
        return self._all_layer_last_state_list

    def _init_hidden(self, batch_size, image_size):
        h_0 = []
        c_0 = []
        for i in range(self.num_layers):
            init_state = self.cell_list[i].init_hidden(batch_size, image_size)
            h_0.append(init_state[0])
            c_0.append(init_state[1])
        return (h_0, c_0)

    def reset_hidden_state(self):
        self._saved_hidden_state = None

    def eval(self, use_saved = True):
        super().eval()
        self.use_saved = use_saved
        self.reset_hidden_state()

    def train(self, mode=True, use_saved = False):
        super().train(mode)
        self.use_saved = use_saved
        self.reset_hidden_state()


    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        # if not (isinstance(kernel_size, tuple) or
        #         (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
        #     raise ValueError('`kernel_size` must be tuple or list of tuples')
        if isinstance(kernel_size, int):
            return (kernel_size, kernel_size)
        elif isinstance(kernel_size, list):
            return [k if isinstance(k, tuple) else (k, k) for k in kernel_size]

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
