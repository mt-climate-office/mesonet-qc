from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping

import numpy as np


@dataclass
class Bits:
    name: str
    order: int
    bits: int
    value: int

    def encode(self):

        if self.value:
            if self.value == -1:
                # If the bit value is -1 (i.e. test cannot be performed),
                # fill the leftmost bit with a 1.
                bit_string = "1".ljust(self.bits, "0")
            else:
                # Otherwise, just use the actuall binary representation,
                # zero padded to fit all assigned bits.
                bit_string = bin(self.value)[2:].zfill(self.bits)
        else:
            # If the QA/QC value is None (i.e. a test hasn't been done yet),
            # fill all bits with one.
            bit_string = "".ljust(self.bits, "1")

        if len(bit_string) > self.bits:
            raise ValueError(
                f"""The QC flag value for {self.name} was greater than its
                allowed number of bits ({self.bits}.) Please check the 
                QC flag and ensure it is valid."""
            )

        return bit_string

    def __lt__(self, other: Bits):
        return self.order < other.order


@dataclass
class BitEncoder:
    qa_step: int = None
    qa_range: int = None
    qa_delta: int = None
    qa_spatial: int = None
    qa_like: int = None
    qa_shared: int = None
    total_bits: int = 32  # How long should the bit string be?
    _bit_list: List[Bits] = field(init=False)
    bit_subsetter: np.ndarray = field(init=False)

    def __post_init__(self):
        self.qa_step = Bits("qa_step", 0, 2, self.qa_step)
        self.qa_range = Bits("qa_range", 1, 2, self.qa_range)
        self.qa_delta = Bits("qa_delta", 2, 2, self.qa_delta)
        self.qa_spatial = Bits("qa_spatial", 3, 2, self.qa_spatial)
        self.qa_like = Bits("qa_like", 4, 8, self.qa_like)
        self.qa_shared = Bits("qa_shared", 5, 8, self.qa_shared)

        self._bit_list = [
            self.qa_step,
            self.qa_range,
            self.qa_delta,
            self.qa_spatial,
            self.qa_like,
            self.qa_shared,
        ]

    def encode(self, as_int: bool = True) -> str | int:
        s_out = ""
        for bit in sorted(self._bit_list):
            s_out = bit.encode() + s_out
        s_out = s_out.zfill(self.total_bits)
        return int(s_out, 2) if as_int else s_out

    @staticmethod
    def code_to_human_readable(code: str, shared: str | None = None):
        code = code[::-1]
        if len(code) > 2 and shared:
            return

        # If all bits are '1'
        if code.count("1") == len(code):
            return "QA/QC not performed for this check."

        # If the leftmost bit is '1' and everything else is '0'.
        if code[0] == "1" and (code.count("0") == len(code) - 1):
            return "Unable to perform QA/QC check. If this is unexpected, check AirTable and metadata."

        return "Finish writing this function@!!!"

    def decode(self, qa_val: str | int) -> Mapping[str, str]:
        if isinstance(qa_val, int):
            qa_val = bin(qa_val)[2:]

        # Reverse so we can use normal indexing
        qa_val = qa_val[::-1]

        cur_idx = 0
        out = {}

        for bit in sorted(self._bit_list):
            end = cur_idx + bit.bits
            code = qa_val[cur_idx:end]
            out[bit.name] = code
            cur_idx = end

        return out


# dat = pd.read_csv("~/Desktop/example.csv")
