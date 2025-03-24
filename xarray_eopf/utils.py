#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

import time


class timeit:
    def __init__(self, label: str | None = None, silent: bool = False):
        self._label = label
        self._silent = silent
        self._t0: float | None = None
        self.time: float | None = None

    def __enter__(self) -> "timeit":
        self._t0 = time.process_time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time = time.process_time() - self._t0
        if not self._silent:
            print(f"{self._label or 'code block'} took {self.time:.3f} seconds")
