import io
import unittest

import numpy as np

from valplat.adapters.livox.livox_reader import LvxReader, PACKET_HEADER_DTYPE
from valplat.util.path import TEST_DATA_PATH


class LvxReaderTest(unittest.TestCase):
    """
    Velodyne extractor unit tests
    """

    def test_extraction(self):
        """Check the extraction from LVX file does not crash"""
        timestamps_s, scan_list = LvxReader().extract(
            TEST_DATA_PATH / "livox_tele15_raw_sample.lvx", 0, 1
        )
        self.assertEqual(
            len(timestamps_s), len(scan_list)
        )  # consistent length of the output arrays
        self.assertGreater(
            min(len(scan) for scan in scan_list), 10000
        )  # scanpoints are present
        np.testing.assert_allclose(
            np.diff(timestamps_s), 0.1, atol=0.002
        )  # default sampling rate is 10Hz

    def test_extraction_dual_return(self):
        """
        Check the extraction from LVX file does not crash for Cartesian Point Cloud with Dual Return.
        Timestamps of IMU packets does not conform to UTC format
        """
        timestamps_s, scan_list = LvxReader().extract(
            TEST_DATA_PATH / "livox_dual_return_sample.lvx", 0, 1
        )
        self.assertEqual(
            len(timestamps_s), len(scan_list)
        )  # consistent length of the output arrays
        self.assertGreater(
            min(len(scan) for scan in scan_list), 10000
        )  # scanpoints are present

    def test_extraction_multi_stream_file(self):
        """Check the extraction from LVX file with multiple streams works"""
        test_data = TEST_DATA_PATH / "livox_multi_raw_sample.lvx"
        expected_streams = "1HDDH3200103281", "1HDDH3200104021", "1PQDH7600100911"
        np.testing.assert_array_equal(
            LvxReader().get_stream_names(test_data), expected_streams
        )
        # check the result from the first two sensors are different
        timestamps1_s, _ = LvxReader(stream_name=expected_streams[0]).extract(
            test_data, 0, 1
        )
        timestamps2_s, _ = LvxReader(stream_name=expected_streams[1]).extract(
            test_data, 0, 1
        )
        self.assertTrue(timestamps1_s != timestamps2_s)

    def test_extraction_corrupted_frame(self):
        """Check the extraction from LVX file corrupted data frame works"""
        test_data = TEST_DATA_PATH / "livox_multi_raw_sample_001.lvx"
        streams = LvxReader().get_stream_names(test_data)
        self.assertEqual(len(streams), 3)
        for stream in streams:
            timestamps, scans = LvxReader(stream_name=stream).extract(test_data)
            self.assertGreater(len(timestamps), 30)
            for scan in scans:
                self.assertGreater(len(scan), 10000)

    def test_read_corrupted_frame(self):
        """Check reading of a corrupted data frame does not crash"""
        read_data_frame = LvxReader()._read_data_frame
        # simulate data frame with unknown datatype
        fstream = io.BytesIO()
        unknown_data_type, next_offset = -1, 200
        fstream.write(np.array([0, next_offset, 300], dtype="Q").tobytes())
        header = np.zeros(1, dtype=PACKET_HEADER_DTYPE)
        header["version"], header["data_type"] = (
            LvxReader.lvx_version,
            unknown_data_type,
        )  # simulate unknown data type
        fstream.write(header.tobytes())
        fstream.seek(0)
        ret = read_data_frame(fstream, 0)
        self.assertEqual(
            ret[0], next_offset
        )  # check the offset is set to the next data frame


if __name__ == "__main__":
    unittest.main()
