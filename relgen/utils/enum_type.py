from enum import Enum


class ModelType(Enum):
    """Type of models.

    - ``AR``: AR Model
    - ``GAN``: GAN Model
    - ``VAE``: VAE Model
    - ``DIFFUSION``: Diffusion Model
    """

    AR = 0
    GAN = 1
    VAE = 2
    DIFFUSION = 3


class SynthesizerType(Enum):
    """Type of synthesizers.

    - ``AR``: AR Synthesizer
    - ``GAN``: GAN Synthesizer
    - ``VAE``: VAE Synthesizer
    - ``DIFFUSION``: Diffusion Synthesizer
    """

    AR = 0
    GAN = 1
    VAE = 2
    DIFFUSION = 3


class VirtualColumnType(Enum):
    """Type of virtual columns.

    - ``NORMAL_ATTR``: Normal attribute column
    - ``INDICATOR``: Indicator column
    - ``FANOUT``: Fanout column
    """

    NORMAL_ATTR = 0
    INDICATOR = 1
    FANOUT = 2


class SynthesisMethod(Enum):
    """Synthesis method.

    - ``SINGLE_MODEL``: Single model
    - ``MULTI_MODEL``: Multi model
    """

    SINGLE_MODEL = 0
    MULTI_MODEL = 1
