import logging
from .pl import PLModel

logger = logging.getLogger("Teachgrav")


class TrueLawModel(PLModel):
    def __init__(self):
        super().__init__(factory=None, G=1.0, power=2.0)
