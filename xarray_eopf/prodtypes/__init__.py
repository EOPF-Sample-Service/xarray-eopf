#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.


def register_product_types():
    from xarray_eopf.prodtype import registry
    from .sentinel1 import register as register_s1
    from .sentinel2 import register as register_s2
    from .sentinel3 import register as register_s3

    register_s1(registry)
    register_s2(registry)
    register_s3(registry)
