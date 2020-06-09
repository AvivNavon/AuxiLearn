from abc import abstractmethod

from torch import nn
from torch.nn.utils import weight_norm


class HyperNet(nn.Module):
    """This module is responsible for taking the losses from all tasks and return a single loss term.
    We can think of this as our learnable loss criterion

    """
    def __init__(self, main_task, input_dim):
        super().__init__()
        self.main_task = main_task
        self.input_dim = input_dim

    def forward(self, losses, outputs=None, labels=None, data=None):
        """

        :param losses: losses form each task. This should be a tensor of size (batch_size, self.input_dim)
        :param outputs: Optional. Parameters model output.
        :param labels: Optional. Target.
        :param data: Optiona. Parameters model input.
        :return:
        """
        pass

    def _init_weights(self):
        pass

    def get_weights(self):
        """

        :return: list of model parameters
        """
        return list(self.parameters())


class MonoHyperNet(HyperNet):
    """Monotonic Hypernets

    """
    def __init__(self, main_task, input_dim, clamp_bias=False):
        super().__init__(main_task=main_task, input_dim=input_dim)
        self.clamp_bias = clamp_bias

    def get_weights(self):
        """

        :return: list of model parameters
        """
        return list(self.parameters())

    @abstractmethod
    def clamp(self):
        pass


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class LinearHyperNet(HyperNet):
    """Linear weights, e.g. \sum_j \alpha_j * l_j

    """
    def __init__(self, main_task, input_dim, skip_connection=False, init_value=1., weight_normalization=True):
        super().__init__(main_task=main_task, input_dim=main_task)

        self.init_value = init_value
        self.skip_connection = skip_connection
        self.linear = nn.Linear(input_dim, 1, bias=False)
        self.weight_normalization = weight_normalization
        self._init_weights()

        if self.weight_normalization:
            self.linear = weight_norm(self.linear)

    def _init_weights(self):
        # init to 1
        # todo: maybe we want 1/num_tasks ?
        self.linear.weight = nn.init.constant_(self.linear.weight, self.init_value)

    def forward(self, losses, outputs=None, labels=None, data=None):
        loss = self.linear(losses).mean()
        if self.skip_connection:
            loss += losses[:, self.main_task].mean()
        return loss


class MonoLinearHyperNet(MonoHyperNet):
    """Linear weights, e.g. \sum_j \alpha_j * l_j

    """
    def __init__(
        self, main_task, input_dim, skip_connection=False, clamp_bias=False, init_value=1., weight_normalization=True
    ):
        super().__init__(main_task=main_task, input_dim=main_task, clamp_bias=clamp_bias)

        self.init_value = init_value
        self.skip_connection = skip_connection
        self.linear = nn.Linear(input_dim, 1, bias=False)
        self._init_weights()

        self.weight_normalization = weight_normalization
        if self.weight_normalization:
            self.linear = weight_norm(self.linear)

    def _init_weights(self):
        # init to 1
        # todo: maybe we want 1/num_tasks ?
        self.linear.weight = nn.init.constant_(self.linear.weight, self.init_value)

    def forward(self, losses, outputs=None, labels=None, data=None):
        loss = self.linear(losses).mean()
        if self.skip_connection:
            loss += losses[:, self.main_task].mean()
        return loss

    def clamp(self):
        """make sure parameters are non-negative

        """
        if self.weight_normalization:
            self.linear.weight_v.data.clamp_(0)
            self.linear.weight_g.data.clamp_(0)
        else:
            self.linear.weight.data.clamp_(0)

        if self.linear.bias is not None and self.clamp_bias:
            self.linear.bias.data.clamp_(0)


class NonlinearHyperNet(HyperNet):

    def __init__(
        self,
        main_task,
        input_dim,
        hidden_sizes=1,
        nonlinearity=None,
        bias=True,
        dropout_rate=0.,
        init_upper=None,
        init_lower=None,
        weight_normalization=True
    ):
        super().__init__(main_task=main_task, input_dim=input_dim)

        assert isinstance(hidden_sizes, (list, int)), "hidden sizes must be int or list"
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        self.nonlinearity = nonlinearity if nonlinearity is not None else nn.Softplus()
        self.dropout = nn.Dropout(dropout_rate)
        self.weight_normalization = weight_normalization

        self.bias = bias
        dims = [self.input_dim] + hidden_sizes + [1]
        self.layers = []

        for j in range(len(dims) - 2):
            self.layers.append(
                self._get_layer(dims[j], dims[j + 1], init_upper=init_upper, init_lower=init_lower, bias=bias)
            )
            self.layers.append(self.nonlinearity)
            self.layers.append(self.dropout)

        self.layers.append(
            self._get_layer(dims[-2], dims[-1], init_upper=init_upper, init_lower=init_lower, bias=False)
        )

        self.net = nn.Sequential(*self.layers)

    def _get_layer(self, input_dim, output_dim, init_upper, init_lower, bias):
        """Create layer with weight normalization

        :param input_dim:
        :param output_dim:
        :param init_upper:
        :param init_lower:
        :param bias:
        :return:
        """
        layer = nn.Linear(input_dim, output_dim, bias=bias)
        self._init_layer(layer, init_upper=init_upper, init_lower=init_lower)
        if self.weight_normalization:
            return weight_norm(layer)
        return layer

    @staticmethod
    def _init_layer(layer, init_lower, init_upper):
        b = init_upper if init_upper is not None else 1.
        a = init_lower if init_lower is not None else 0.
        if isinstance(layer, nn.Linear):
            layer.weight = nn.init.uniform_(layer.weight, b=b, a=a)
            if layer.bias is not None:
                layer.bias = nn.init.constant_(layer.bias, 0.)

    def forward(self, losses, outputs=None, labels=None, data=None):
        main_loss = losses[:, self.main_task].mean()
        return self.net(losses).mean() + main_loss


class MonoNonlinearHyperNet(MonoHyperNet):

    def __init__(
        self,
        main_task,
        input_dim,
        hidden_sizes=1,
        nonlinearity=None,
        bias=True,
        dropout_rate=0.,
        init_upper=None,
        init_lower=None,
        weight_normalization=True
    ):
        super().__init__(main_task=main_task, input_dim=input_dim)

        assert isinstance(hidden_sizes, (list, int)), "hidden sizes must be int or list"
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        self.nonlinearity = nonlinearity if nonlinearity is not None else nn.Softplus()
        self.dropout = nn.Dropout(dropout_rate)
        self.weight_normalization = weight_normalization

        if isinstance(self.nonlinearity, Identity):
            bias = False

        self.bias = bias
        dims = [self.input_dim] + hidden_sizes + [1]
        self.layers = []

        for j in range(len(dims) - 2):
            self.layers.append(
                self._get_layer(dims[j], dims[j + 1], init_upper=init_upper, init_lower=init_lower, bias=bias)
            )
            self.layers.append(self.nonlinearity)
            self.layers.append(self.dropout)

        self.layers.append(
            self._get_layer(dims[-2], dims[-1], init_upper=init_upper, init_lower=init_lower, bias=False)
        )

        self.net = nn.Sequential(*self.layers)

    def _get_layer(self, input_dim, output_dim, init_upper, init_lower, bias):
        """Create layer with weight normalization

        :param input_dim:
        :param output_dim:
        :param init_upper:
        :param init_lower:
        :param bias:
        :return:
        """
        layer = nn.Linear(input_dim, output_dim, bias=bias)
        self._init_layer(layer, init_upper=init_upper, init_lower=init_lower)
        if self.weight_normalization:
            return weight_norm(layer)
        return layer

    @staticmethod
    def _init_layer(layer, init_lower, init_upper):
        b = init_upper if init_upper is not None else 1.
        a = init_lower if init_lower is not None else 0.
        if isinstance(layer, nn.Linear):
            layer.weight = nn.init.uniform_(layer.weight, b=b, a=a)
            if layer.bias is not None:
                layer.bias = nn.init.constant_(layer.bias, 0.)

    def forward(self, losses, outputs=None, labels=None, data=None):
        main_loss = losses[:, self.main_task].mean()
        return self.net(losses).mean() + main_loss

    def clamp(self):
        for l in self.net:
            if isinstance(l, nn.Linear):
                if self.weight_normalization:
                    l.weight_v.data.clamp_(0)
                    l.weight_g.data.clamp_(0)
                else:
                    l.weight.data.clamp_(0)

                if l.bias is not None and self.clamp_bias:
                    l.bias.data.clamp_(0)


class NoFCCNNHyperNet(HyperNet):

    # NYU Input shape is (3, 288, 384)

    def __init__(
        self, main_task, reduction='mean', input_channels=3, init_upper=.1, init_lower=0, weight_normalization=False
    ):
        super().__init__(input_dim=-1, main_task=main_task)

        self.main_task = main_task
        assert reduction in ['mean', 'sum']
        self.reduction = reduction
        self.weight_normalization = weight_normalization

        self.conv = nn.Sequential(
            self._get_layer(
                input_channels, 8, kernel_size=3, stride=1, padding=1, init_lower=init_lower, init_upper=init_upper
            ),
            nn.Softplus(),
            nn.AvgPool2d(kernel_size=2, padding=0),
            self._get_layer(8, 16, kernel_size=3, stride=1, padding=1, init_lower=init_lower, init_upper=init_upper),
            nn.Softplus(),
            nn.AvgPool2d(kernel_size=2, padding=0),
            self._get_layer(16, 32, kernel_size=3, stride=1, padding=1, init_lower=init_lower, init_upper=init_upper),
            nn.Softplus(),
            nn.AvgPool2d(kernel_size=2, padding=0),
            self._get_layer(32, 1, kernel_size=1, bias=False),
        )

    def _get_layer(
        self,
        input_channels,
        output_channel,
        init_upper=None,
        init_lower=None,
        bias=True,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        layer = nn.Conv2d(
            input_channels,
            output_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self._init_layer(layer, init_upper=init_upper, init_lower=init_lower)
        if self.weight_normalization:
            return weight_norm(layer)
        return layer

    @staticmethod
    def _init_layer(layer, init_lower, init_upper):
        b = init_upper if init_upper is not None else 1.
        a = init_lower if init_lower is not None else 0.
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            layer.weight = nn.init.uniform_(layer.weight, b=b, a=a)
            if layer.bias is not None:
                layer.bias = nn.init.constant_(layer.bias, 0.)

    def forward(self, losses, outputs=None, labels=None, data=None):
        main_loss = .0
        if self.main_task is not None:
            # (bs, tasks, h, w)
            if self.reduction == 'mean':
                main_loss = losses[:, self.main_task, :, :].mean(dim=(1, 2)).mean(0)
            else:
                main_loss = losses[:, self.main_task, :, :].sum(dim=(1, 2)).mean(0)

        # self.conv(losses) is of shape (bs, 1, 36, 48)
        img_loss = self.conv(losses).mean((1, 2, 3))
        return img_loss.mean() + main_loss


class MonoNoFCCNNHyperNet(MonoHyperNet):

    # NYU Input shape is (3, 288, 384)

    def __init__(
        self, main_task, reduction='mean', init_upper=.1, init_lower=0., input_channels=3, weight_normalization=False
    ):
        super().__init__(input_dim=-1, main_task=main_task)

        self.main_task = main_task
        assert reduction in ['mean', 'sum']
        self.reduction = reduction
        self.weight_normalization = weight_normalization

        self.conv = nn.Sequential(
            self._get_layer(
                input_channels, 8, kernel_size=3, stride=1, padding=1, init_lower=init_lower, init_upper=init_upper
            ),
            nn.Softplus(),
            nn.AvgPool2d(kernel_size=2, padding=0),
            self._get_layer(8, 16, kernel_size=3, stride=1, padding=1, init_lower=init_lower, init_upper=init_upper),
            nn.Softplus(),
            nn.AvgPool2d(kernel_size=2, padding=0),
            self._get_layer(16, 1, kernel_size=1, bias=False, init_lower=init_lower, init_upper=init_upper),
        )

    def _get_layer(
        self,
        input_channels,
        output_channel,
        init_upper=None,
        init_lower=None,
        bias=True,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        layer = nn.Conv2d(
            input_channels,
            output_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self._init_layer(layer, init_upper=init_upper, init_lower=init_lower)
        if self.weight_normalization:
            return weight_norm(layer)
        return layer

    @staticmethod
    def _init_layer(layer, init_lower, init_upper):
        b = init_upper if init_upper is not None else 1.
        a = init_lower if init_lower is not None else 0.
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            layer.weight = nn.init.uniform_(layer.weight, b=b, a=a)
            if layer.bias is not None:
                layer.bias = nn.init.constant_(layer.bias, 0.)

    def forward(self, losses, outputs=None, labels=None, data=None):
        main_loss = .0
        if self.main_task is not None:
            # (bs, tasks, h, w)
            if self.reduction == 'mean':
                main_loss = losses[:, self.main_task, :, :].mean(dim=(1, 2)).mean(0)
            else:
                main_loss = losses[:, self.main_task, :, :].sum(dim=(1, 2)).mean(0)

        # self.conv(losses) is of shape (bs, 1, 36, 48)
        img_loss = self.conv(losses).mean((1, 2, 3)).mean()

        return img_loss + main_loss

    def clamp(self):
        for l in self.conv:
            if isinstance(l, nn.Conv2d):
                if self.weight_normalization:
                    l.weight_v.data.clamp_(0)
                    l.weight_g.data.clamp_(0)
                else:
                    l.weight.data.clamp_(0)

                if l.bias is not None and self.clamp_bias:
                    l.bias.data.clamp_(0)
