from dataclasses import dataclass

import pandas as pd

BIT_MAP = {-1: "10", 0: "00", 1: "01"}


@dataclass
class bits:

    qa_step: int = None
    qa_range: int = None
    qa_delta: int = None
    qa_shared: int = None

    def encode(self):
        pass

    def decode(self):
        pass


def encode_qc_bits(checked: pd.DataFrame):
    pass


def decode_qc_bits():
    pass
