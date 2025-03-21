#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from contextlib import contextmanager

import time


@contextmanager
def timeit(label: str | None = None, silent: bool = False):
    t0 = time.process_time()
    result = dict(t0=t0)
    try:
        yield result
    finally:
        t1 = time.process_time()
        dt = t1 - t0
        result.update(t1=t1, dt=dt)
        if not silent:
            print(f"{label or 'code block'} took {dt:.3f} seconds")
