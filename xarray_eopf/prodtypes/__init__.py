def register_product_types():
    from xarray_eopf.prodtype import registry
    from .sentinel1 import register_s1_product_types
    from .sentinel2 import register_s2_product_types
    from .sentinel3 import register_s3_product_types

    register_s1_product_types(registry)
    register_s2_product_types(registry)
    register_s3_product_types(registry)
