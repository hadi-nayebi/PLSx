"""Class Architecture defines blueprint for building layers of a model."""
from collections.abc import Generator
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Extra
from torch import Tensor, transpose
from torch.nn import (
    ELU,
    Conv1d,
    ConvTranspose1d,
    Flatten,
    Linear,
    LogSoftmax,
    MaxPool1d,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
    Softmax,
    Tanh,
    Unflatten,
)
from torch.nn.init import kaiming_normal_, xavier_normal_

from Plsx.dataloader.utils import read_json


def get_activation(activation: str) -> Module:
    """Return the activation function."""
    if activation == "ReLU":
        return ReLU()
    elif activation == "ELU":
        return ELU()
    elif activation == "Sigmoid":
        return Sigmoid()
    elif activation == "Tanh":
        return Tanh()
    elif activation == "Softmax":
        return Softmax(dim=1)
    elif activation == "LogSoftmax":
        return LogSoftmax(dim=1)
    else:
        raise ValueError(f"{activation} is not a valid activation.")


class Layer(BaseModel):
    """Layer class defines blueprint for building layers of a model."""

    type: str

    class Config:
        extra = Extra.allow

    def make(self) -> Generator[Module]:
        """Make the layer."""
        raise NotImplementedError

    def init_weights(self) -> None:
        # check if m has init attributes
        if hasattr(self, "init"):
            if self.init == "xavier":
                xavier_normal_(self.layer.weight)
            elif self.init == "he":
                kaiming_normal_(self.layer.weight)


class _Linear(Layer):
    """Linear class."""

    in_features: int
    out_features: int
    init: Optional[str] = "xavier"
    bias: Optional[bool] = False
    activation: Optional[str] = "Tanh"

    def make(self) -> Generator[Module]:
        """Make the linear layer. As a generator yields layers."""
        assert self.type == "Linear"
        self.layer = Linear(self.in_features, self.out_features, bias=self.bias)
        self.init_weights()
        yield self.layer
        self.activation_layer = get_activation(self.activation)
        yield self.activation_layer


class _Conv1d(Layer):
    """Conv1d class."""

    in_features: int
    out_features: int
    kernel: int
    padding: Optional[Union[int, str]] = 0
    init: Optional[str] = "xavier"
    activation: Optional[str] = "Tanh"

    def make(self) -> Generator[Module]:
        """Make the conv1d layer. As a generator yields layers."""
        assert self.type == "Conv1d"
        self.layer = Conv1d(self.in_features, self.out_features, self.kernel, padding=self.padding)
        self.init_weights()
        yield self.layer
        self.activation_layer = get_activation(self.activation)
        yield self.activation_layer


class _MaxPool1d(Layer):
    """MaxPool1d class."""

    kernel: int

    def make(self) -> Generator[Module]:
        """Make the maxpool1d layer. As a generator yields layers."""
        assert self.type == "MaxPool1d"
        self.layer = MaxPool1d(self.kernel)
        yield self.layer


class _Flatten(Layer):
    """Flatten class."""

    def make(self) -> Generator[Module]:
        """Make the flatten layer. As a generator yields layers."""
        assert self.type == "Flatten"
        self.layer = Flatten()
        yield self.layer


class _Unflatten(Layer):
    """Unflatten class."""

    in_features: int
    out_features: int

    def make(self) -> Generator[Module]:
        """Make the unflatten layer. As a generator yields layers."""
        assert self.type == "Unflatten"
        self.layer = Unflatten(1, (self.in_features, self.out_features))
        yield self.layer


class _ConvTranspose1d(Layer):
    """ConvTranspose1d class."""

    in_features: int
    out_features: int
    kernel: int
    padding: Optional[Union[int, str]] = 0
    init: Optional[str] = "he"
    activation: Optional[str] = "ReLU"

    def make(self) -> Generator[Module]:
        """Make the convtranspose1d layer. As a generator yields layers."""
        assert self.type == "ConvTranspose1d"
        self.layer = ConvTranspose1d(
            self.in_features, self.out_features, self.kernel, padding=self.padding
        )
        self.init_weights()
        yield self.layer
        self.activation_layer = get_activation(self.activation)
        yield self.activation_layer


class Unit(BaseModel):
    """Unit class defines blueprint for building units of a model."""

    layers: list[Layer] = []
    unit: Module = None

    class Config:
        arbitrary_types_allowed = True

    def build(self, config: list[dict[str, Union[str, int, bool]]]) -> None:
        """Creates layers by type and appends to its layers list."""
        if len(self.layers) == 0:
            if len(config) > 0:
                for layer in config:
                    self.layers.append(self.layer_maker(layer))
            else:
                raise ValueError("Config is empty.")
        else:
            raise Warning("This unit has already been made.")

    def make(self) -> Generator[Module]:
        """Make the unit. As a generator yields layers."""
        for layer in self.layers:
            yield from layer.make()

    def add_layer(self, layer: Layer) -> None:
        """Add layer to the unit."""
        # needs extra work
        raise NotImplementedError
        # self.layers.append(layer)

    def insert_layer(self, layer: Layer, index: int) -> None:
        """Insert layer at index."""
        # needs extra work
        raise NotImplementedError
        # self.layers.insert(index, layer)

    def input_shape(self) -> int:
        """Return the input shape."""
        return self.layers[0].in_features

    def output_shape(self) -> int:
        """Return the output shape."""
        return self.layers[-1].out_features

    @staticmethod
    def layer_maker(layer: dict[str, Union[str, int, bool]]) -> Layer:
        """Make the layer."""
        if layer["type"] == "Linear":
            return _Linear(**layer)
        elif layer["type"] == "Conv1d":
            return _Conv1d(**layer)
        elif layer["type"] == "MaxPool1d":
            return _MaxPool1d(**layer)
        elif layer["type"] == "Flatten":
            return _Flatten(**layer)
        elif layer["type"] == "Unflatten":
            return _Unflatten(**layer)
        elif layer["type"] == "ConvTranspose1d":
            return _ConvTranspose1d(**layer)
        else:
            raise ValueError(f"{layer['type']} is not a valid layer.")

    def forward(self, **kwargs) -> Tensor:
        """Forward pass."""
        x = self.unit.forward(kwargs["x"])
        return x


class Vectorizer(Unit):
    """Vectorizer class."""

    def forward(self, **kwargs) -> Tensor:
        """Forward pass."""
        x = kwargs["x"].reshape((-1, kwargs["d0"]))
        return self.unit.forward(x)


class Devectorizer(Unit):
    """Devectorizer class."""


class Encoder(Unit):
    """Encoder class."""

    def forward(self, **kwargs) -> Tensor:
        """Forward pass."""
        x = transpose(kwargs["x"].reshape(-1, kwargs["w"], kwargs["d1"]), 1, 2)
        return self.unit.forward(x)


class Decoder(Unit):
    """Decoder class."""

    def forward(self, **kwargs) -> Tensor:
        """Forward pass."""
        x = self.unit.forward(kwargs["x"])
        x = transpose(x, 1, 2).reshape(-1, kwargs["d1"])
        return x


class SSDecoder(Unit):
    """SSDecoder class."""

    def forward(self, **kwargs) -> Tensor:
        """Forward pass."""
        x = self.unit.forward(kwargs["x"])
        x = transpose(x, 1, 2).reshape(-1, kwargs["ds"])
        return x


class Classifier(Unit):
    """Classifier class."""


class Discriminator(Unit):
    """Discriminator class."""


class Architecture(object):
    """
    The Architecture object provides the model arch params.
    """

    def __init__(self) -> None:
        self.name: str = None
        self.type: str = None
        self.w: int = None
        self.components: dict[str, Unit] = {}
        self.connections: list[tuple[str, str]] = []
        self.is_built: bool = False

    def build(self, src: Union[str, Path]) -> None:
        """build from file."""
        if isinstance(src, str):
            src = Path(src)
        if src.exists():
            arch = read_json(src)
            assert self.is_valid_architecture(arch)
            self.name = arch["name"]
            self.type = arch["type"]
            self.w = arch["window"]
            self.connections = [tuple(item) for item in arch["connections"]]
            # build components
            for key, item in arch["components"].items():
                self.components[key] = self.build_units(key, item)

    def get_model(self) -> dict[str, Module]:
        """build the model."""
        if not self.is_built:
            for _, unit in self.components.items():
                unit.unit = Sequential(*unit.make())
            self.validate_connections()
            self.is_built = True

    def validate_connections(self) -> None:
        """validate connections."""
        assert len(self.connections) > 0
        for src, dst in self.connections:
            if not (src == "input" or dst == "output"):
                assert src in self.components.keys()
                assert dst in self.components.keys()
                assert self.components[src].output_shape() == self.components[dst].input_shape()

    @property
    def d1(self) -> int:
        """Return the arch params."""
        d1 = None
        if "vectorizer" in self.components.keys():
            d1 = self.components.get("vectorizer").output_shape()
        if "devectorizer" in self.components.keys():
            assert self.components.get("devectorizer").input_shape() == d1
        return d1

    @property
    def dn(self) -> int:
        dn = None
        if "encoder" in self.components.keys():
            dn = self.components.get("encoder").output_shape()
        if "decoder" in self.components.keys():
            assert self.components.get("decoder").input_shape() == dn
        return dn

    @property
    def d0(self) -> int:
        d0 = None
        if "vectorizer" in self.components.keys():
            d0 = self.components.get("vectorizer").input_shape()
        if "devectorizer" in self.components.keys():
            assert self.components.get("devectorizer").output_shape() == d0
        return d0

    @property
    def ds(self) -> int:
        ds = None
        if "ss_decoder" in self.components.keys():
            ds = self.components.get("ss_decoder").output_shape()
        return ds

    @staticmethod
    def build_units(type: str, config: list[dict[str, Union[str, int, bool]]]) -> Unit:
        """build units from config."""
        assert len(config) > 0
        if type == "vectorizer":
            assert len(config) == 1
            unit = Vectorizer()
        elif type == "devectorizer":
            assert len(config) == 1
            unit = Devectorizer()
        elif type == "encoder":
            unit = Encoder()
        elif type == "decoder":
            unit = Decoder()
        elif type == "ss_decoder":
            unit = SSDecoder()
        elif type == "classifier":
            unit = Classifier()
        elif type == "discriminator":
            unit = Discriminator()
        else:
            raise ValueError("Unknown unit type.")

        unit.build(config)
        return unit

    @staticmethod
    def is_valid_architecture(arch: dict) -> bool:
        """Check if the architecture is valid."""
        assert "name" in arch
        assert "type" in arch
        assert "components" in arch
        assert "connections" in arch
        assert isinstance(arch["components"], dict)
        assert isinstance(arch["connections"], list)
        return True

    def forward(self, x: Tensor, connections: list[tuple[str, str]] = None) -> Tensor:
        """forward."""
        assert isinstance(x, Tensor)
        assert x.dim() == 3
        assert x.shape[1] == self.w
        assert x.shape[2] == self.d0
        assert self.is_built
        input_args = {"x": x, "d0": self.d0, "d1": self.d1, "w": self.w, "ds": self.ds}
        vals = {}
        if connections is None:
            connections = self.connections
        for connection in connections:
            src, dst = connection
            if src == "input":
                vals[dst] = self.components[dst].forward(**input_args)
            elif dst == "output":
                vals[dst] = True
            else:
                input_args["x"] = vals[src]
                vals[dst] = self.components[dst].forward(**input_args)
        return vals
