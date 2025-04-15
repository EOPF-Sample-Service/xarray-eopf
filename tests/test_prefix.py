#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

from xarray_eopf.prefix import (
    get_common_prefix,
    get_common_string_prefix,
    get_unique_short_sequences,
    get_unique_short_strings,
)


class CommonPrefixTest(TestCase):
    def test_same(self):
        self.assertEqual(
            [],
            get_common_prefix("satellite", "satellite"),
        )

    def test_different(self):
        self.assertEqual(["s", "a", "t"], get_common_prefix("satellite", "saturation"))


class CommonStringPrefixTest(TestCase):
    def test_same(self):
        self.assertEqual("", get_common_string_prefix("satellite", "satellite"))

    def test_different(self):
        self.assertEqual("sat", get_common_string_prefix("satellite", "saturation"))


class UniqueShortSequencesTest(TestCase):
    def test_all_same(self):
        seq1 = tuple("S01SIWGRD_20240201T164915_0025_A299_750E_065517_VH".split("_"))
        seq2 = tuple("S01SIWGRD_20240201T164915_0025_A299_750E_065517_VH".split("_"))
        self.assertEqual(
            {
                (
                    "S01SIWGRD",
                    "20240201T164915",
                    "0025",
                    "A299",
                    "750E",
                    "065517",
                    "VH",
                ): (
                    "S01SIWGRD",
                    "20240201T164915",
                    "0025",
                    "A299",
                    "750E",
                    "065517",
                    "VH",
                )
            },
            get_unique_short_sequences([seq1, seq2]),
        )

    def test_all_different(self):
        seq1 = "S01SIWGRD_20240201T164915_0025_A299_750E_065517_VH".split("_")
        seq2 = "S01SIWGRD_20240201T164915_0025_A299_750E_065517_VV".split("_")
        self.assertEqual(
            {
                (
                    "S01SIWGRD",
                    "20240201T164915",
                    "0025",
                    "A299",
                    "750E",
                    "065517",
                    "VH",
                ): ("VH",),
                (
                    "S01SIWGRD",
                    "20240201T164915",
                    "0025",
                    "A299",
                    "750E",
                    "065517",
                    "VV",
                ): ("VV",),
            },
            get_unique_short_sequences([seq1, seq2]),
        )


class UniqueShortStringsTest(TestCase):
    def test_all_same(self):
        self.assertEqual(
            {"satellite": "satellite"},
            get_unique_short_strings(["satellite", "satellite", "satellite"]),
        )

    def test_all_different(self):
        self.assertEqual(
            {
                "satellite": "ellite",
                "satisfy": "isfy",
                "saturation": "uration",
            },
            get_unique_short_strings(["satellite", "saturation", "satisfy"]),
        )
