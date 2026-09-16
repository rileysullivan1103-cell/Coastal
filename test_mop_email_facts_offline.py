#!/usr/bin/env python3
"""Offline checks for mop_email_facts.py.

The fixture is built so every answer is known by construction: a 500-hour
series whose usable hours for one variable are placed at chosen indices, a
second variable left entirely fill, and a third dropped from the frame
altogether. Nothing here asserts against a value copied out of the module.
"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import mop_email_facts as mef

HOURS = 500
USABLE_INDEX = [10, 11, 12, 400]


def make_frame():
    time = pd.date_range("1999-12-31", periods=HOURS, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "time": time,
        "wave_hs_m": 1.5,
        "wave_sxy": np.nan,
        "wave_sxx": np.nan,
        "product": ["hindcast"] * 400 + ["nowcast"] * (HOURS - 400),
        "mop_id": "SC130",
    })
    frame.loc[USABLE_INDEX, "wave_sxy"] = 0.5
    return frame


class Facts(unittest.TestCase):
    def setUp(self):
        self.frame = make_frame()

    def test_point_id_read_from_the_file(self):
        self.assertEqual(mef.point_id(self.frame), "SC130")

    def test_two_point_ids_refuse(self):
        mixed = self.frame.copy()
        mixed.loc[0, "mop_id"] = "SC129"
        with self.assertRaises(SystemExit):
            mef.point_id(mixed)

    def test_span_is_the_first_and_last_hour(self):
        first, last = mef.span(self.frame)
        self.assertEqual(first, self.frame["time"].iloc[0])
        self.assertEqual(last, self.frame["time"].iloc[-1])
        self.assertEqual(int((last - first) / pd.Timedelta(hours=1)),
                         HOURS - 1)

    def test_products_split_where_the_fixture_splits_them(self):
        table = mef.by_product(self.frame)
        self.assertEqual(table.loc["hindcast", "count"], 400)
        self.assertEqual(table.loc["nowcast", "count"], HOURS - 400)

    def test_camel_case_variable_finds_snake_case_column(self):
        self.assertEqual(mef.column_for(self.frame, "waveSxy"), "wave_sxy")
        self.assertEqual(mef.column_for(self.frame, "waveHs"), None)

    def test_usable_count_and_bracket_match_the_planted_hours(self):
        present, total, first, last = mef.populated(self.frame, "waveSxy")
        self.assertEqual(present, len(USABLE_INDEX))
        self.assertEqual(total, HOURS)
        self.assertEqual(first, self.frame["time"].iloc[min(USABLE_INDEX)])
        self.assertEqual(last, self.frame["time"].iloc[max(USABLE_INDEX)])

    def test_all_fill_column_reads_zero_not_absent(self):
        present, total, first, last = mef.populated(self.frame, "waveSxx")
        self.assertEqual(present, 0)
        self.assertEqual(total, HOURS)
        self.assertIsNone(first)
        self.assertIsNone(last)

    def test_dropped_column_reads_absent_not_zero(self):
        """A column the audit removed must not look like one full of fill.

        Both mean 'no usable data', but only one of them means CDIP never
        served anything -- and the email says different things about each.
        """
        without = self.frame.drop(columns=["wave_sxx"])
        present, total, _, _ = mef.populated(without, "waveSxx")
        self.assertIsNone(present)
        self.assertEqual(total, HOURS)

    def test_url_carries_the_point_variable_and_chunk(self):
        url = mef.request_url("SC130", "waveSxy", chunk=20000)
        self.assertTrue(url.startswith("https://thredds.cdip.ucsd.edu/"))
        self.assertIn("SC130_hindcast.nc.ascii?", url)
        self.assertIn("waveSxy[0:1:19999]", url)

    def test_url_point_follows_the_file_not_a_default(self):
        other = self.frame.copy()
        other["mop_id"] = "SC131"
        url = mef.request_url(mef.point_id(other), "waveSxy")
        self.assertIn("SC131_", url)
        self.assertNotIn("SC130", url)

    def test_find_csv_errors_when_nothing_was_pulled(self):
        start = os.getcwd()
        with tempfile.TemporaryDirectory() as empty:
            os.chdir(empty)
            try:
                with self.assertRaises(SystemExit):
                    mef.find_csv()
            finally:
                os.chdir(start)


if __name__ == "__main__":
    unittest.main()
