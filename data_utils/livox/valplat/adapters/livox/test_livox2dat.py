import numpy as np
import unittest
import tempfile
from pathlib import Path

from valplat.adapters.livox.livox2dat import lvx_to_dat, LvxBinWrapper
from valplat.util.path import TEST_DATA_PATH


class Lvx2DatTest(unittest.TestCase):
    def test_lvx_to_dat(self):
        """Check the conversion to LE dat works"""
        with tempfile.TemporaryDirectory() as out_dir:
            lvx_to_dat(
                TEST_DATA_PATH / "livox_multi_raw_sample.lvx",
                out_dir,
                time_bouds=(0, 10),
            )
            self.assertEqual(len(list(Path(out_dir).glob("*.dat"))), 1)

    def test_lvx_to_dat_roboauto(self):
        with tempfile.TemporaryDirectory() as out_dir:
            lvx_to_dat(
                TEST_DATA_PATH / "livox_roboauto.bin",
                out_dir,
                roboauto_bin=True,
                time_bouds=(0, np.inf),
            )
            self.assertEqual(len(list(Path(out_dir).glob("*.dat"))), 1)

            wrp = LvxBinWrapper(TEST_DATA_PATH / "livox_roboauto.bin")
            scan_list = wrp.get_scan_list("Scan", 0, np.inf)

            np.testing.assert_array_almost_equal(
                [
                    0.0,
                    0.09984040260314941,
                    0.19968008995056152,
                    0.29952001571655273,
                    0.39936041831970215,
                    0.49920010566711426,
                    0.5990400314331055,
                    0.6988801956176758,
                    0.7987203598022461,
                    0.8985848426818848,
                    0.9945850372314453,
                ],
                [t[0] for t in scan_list],
            )

            np.testing.assert_array_almost_equal(
                [7808, 7492, 7711, 7534, 7575, 7559, 7548, 7712, 7658, 7481, 6997],
                [len(t[1]) for t in scan_list],
            )

            np.testing.assert_array_almost_equal(
                [
                    53.308,
                    53.404,
                    53.425,
                    46.646,
                    54.034,
                    46.217,
                    46.348,
                    46.231,
                    46.449,
                    46.451,
                ],
                scan_list[0][1]["x"][:10],
            )


if __name__ == "__main__":
    unittest.main()
