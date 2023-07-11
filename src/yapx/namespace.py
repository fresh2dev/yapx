from argparse import Namespace as _Namespace
from typing import Any, Dict


class Namespace(_Namespace):
    def to_dict(self, include_all: bool = False) -> Dict[str, Any]:
        if include_all:
            return vars(self)
        return {k: v for k, v in vars(self).items() if k and k[0].isalpha()}
