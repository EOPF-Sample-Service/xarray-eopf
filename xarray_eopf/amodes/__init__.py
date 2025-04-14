#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.


def register_analysis_modes():
    from xarray_eopf.amode import AnalysisMode

    from .sentinel1 import register as register_s1_analysis_modes
    from .sentinel2 import register as register_s2_analysis_modes
    from .sentinel3 import register as register_s3_analysis_modes

    register_s1_analysis_modes(AnalysisMode.registry)
    register_s2_analysis_modes(AnalysisMode.registry)
    register_s3_analysis_modes(AnalysisMode.registry)
