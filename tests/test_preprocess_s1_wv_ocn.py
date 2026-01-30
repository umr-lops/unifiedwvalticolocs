import datetime
import os

import xarray as xr

import unifiedwvalticolocs
from unifiedwvalticolocs.unified_coloc_WV_alti_cmems_or_cci import preprocess_wv_s1_ocn

# Replace 'your_module' with the actual name of the python file containing the function
# from your_module import preprocess_wv_s1_ocn


def test_preprocess_wv_s1_ocn():
    """
    Unit test for preprocess_wv_s1_ocn using a real S1 WV OCN file.
    """
    # 1. Path to the test file
    test_file = os.path.join(
        os.path.dirname(unifiedwvalticolocs.__file__),
        "data4tests",
        "s1a-wv2-ocn-vv-20260112t045817-20260112t045820-062729-07ddb1-036.nc",
    )
    # Check if file exists to avoid confusing error messages
    assert os.path.exists(test_file), f"Test file not found at {test_file}"

    # 2. Open the dataset
    # We use chunks={} or open_dataset to ensure encoding['source'] is populated
    ds_raw = xr.open_dataset(test_file)

    # 3. Apply the preprocessing function

    ds_processed = preprocess_wv_s1_ocn(ds_raw)

    # 4. Assertions

    # A. Check if the new dimension/coordinate 'time_sar' exists
    assert "time_sar" in ds_processed.dims
    assert "time_sar" in ds_processed.coords

    # B. Check if the date parsing is correct
    # The function uses index [5] of the filename split by '-':
    # [4] is 20260112t045817, [5] is 20260112t045820
    expected_date = datetime.datetime(2026, 1, 12, 4, 58, 20)
    actual_date = (
        ds_processed.time_sar.values[0].astype("M8[ms]").astype(datetime.datetime)
    )

    # Note: Depending on numpy/xarray version, you might need to handle np.datetime64 conversion
    assert actual_date.strftime("%Y%m%d%H%M%S") == expected_date.strftime(
        "%Y%m%d%H%M%S"
    )

    # C. Check if variables are filtered correctly
    # 'oswLon' is in the 'to_keep_vars' list
    assert "oswLon" in ds_processed.data_vars
    # Ensure a random variable NOT in the list is removed (if it existed in raw)
    # (Assuming 'oswRaSize' was a dimension or variable in the original file)
    assert "oswRaSize" not in ds_processed.dims

    # D. Check if squeezed dimensions are gone
    assert "oswRaSize" not in ds_processed.dims
    assert "oswAzSize" not in ds_processed.dims

    # E. Check if scalar variables were expanded to include time_sar
    # Pick a variable that is normally a 2D/Scalar in S1 OCN but should now have time_sar
    for var in ds_processed.data_vars:
        assert "time_sar" in ds_processed[var].dims

    print("\nUnit test passed successfully!")


if __name__ == "__main__":
    # This allows running the test directly via 'python test_preprocess.py'
    test_preprocess_wv_s1_ocn()
