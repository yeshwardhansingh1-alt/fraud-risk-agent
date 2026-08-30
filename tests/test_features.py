import pytest
import pandas as pd
import numpy as np
from features.entity_graph import _cumulative_unique

def test_cumulative_unique():
    # Verify no data leakage in cumulative unique graph features
    df = pd.DataFrame({
        "TransactionDT": [1, 2, 3, 4],
        "DeviceInfo": ["Dev1", "Dev1", "Dev2", "Dev1"],
        "card1": ["C1", "C2", "C1", "C2"]
    })
    
    out = _cumulative_unique(df.copy(), "DeviceInfo", "card1", "cards_sharing_device")
    
    # Dev1 sees C1 at t=1 (count=1)
    # Dev1 sees C2 at t=2 (count=2)
    # Dev2 sees C1 at t=3 (count=1)
    # Dev1 sees C2 at t=4 (already seen, count should still be 2)
    
    expected = [1, 2, 1, 2]
    np.testing.assert_array_equal(out["cards_sharing_device"].values, expected)
