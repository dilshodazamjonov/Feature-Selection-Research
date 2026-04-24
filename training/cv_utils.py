import numpy as np
import pandas as pd
from typing import Iterator, Tuple
from sklearn.model_selection import TimeSeriesSplit


def _to_1d_proba(values):
    """Convert 2D probability array to 1D, taking the positive class column."""
    values = np.asarray(values)
    if values.ndim == 2:
        if values.shape[1] == 1:
            values = values[:, 0]
        else:
            values = values[:, 1]
    return values.reshape(-1)


class GroupedTimeSeriesSplit:
    """
    Time-series CV that keeps identical time values in the same fold.

    Standard ``TimeSeriesSplit`` works on row order, which can split a single
    timestamp across train and validation when multiple rows share the same
    period. This wrapper first splits unique time groups, then expands each
    split back to row indices.
    """

    def __init__(self, n_splits: int = 5, gap: int = 0):
        self.n_splits = n_splits
        self.gap = gap

    def split(self, time_values) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        time_values = np.asarray(time_values)

        if time_values.ndim != 1:
            raise ValueError("time_values must be one-dimensional")
        if pd.isna(time_values).any():
            raise ValueError("time_values contains missing values")

        unique_times = np.unique(time_values)
        if len(unique_times) <= self.n_splits:
            raise ValueError(
                f"Need more unique time groups than n_splits. "
                f"Got {len(unique_times)} unique groups for n_splits={self.n_splits}."
            )

        splitter = TimeSeriesSplit(n_splits=self.n_splits)

        for train_row_idx, val_row_idx in splitter.split(np.arange(len(time_values))):
            val_start_time = time_values[val_row_idx[0]]
            val_end_time = time_values[val_row_idx[-1]]

            val_start_pos = np.searchsorted(unique_times, val_start_time)
            val_end_pos = np.searchsorted(unique_times, val_end_time, side="right")
            train_end_pos = max(val_start_pos - self.gap, 0)

            train_times = unique_times[:train_end_pos]
            val_times = unique_times[val_start_pos:val_end_pos]

            train_idx = np.flatnonzero(np.isin(time_values, train_times))
            val_idx = np.flatnonzero(np.isin(time_values, val_times))

            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class SlidingWindowSplit:
    """
    Sliding window time series cross-validation.
    
    Unlike TimeSeriesSplit (expanding window), this uses a fixed-size sliding window:
    - Fold 1: train on [0:window_size], validate on [window_size:window_size*2]
    - Fold 2: train on [window_size:window_size*2], validate on [window_size*2:window_size*3]
    - etc.
    
    This ensures each fold has equal training size and tests on sequential time periods.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        window_size: int = None,
        test_size: int = None,
        gap: int = 0
    ):
        """
        Parameters
        ----------
        n_splits : int
            Number of splits (folds)
        window_size : int, optional
            Size of training window. If None, calculated as len(X) / (n_splits + 1)
        test_size : int, optional
            Size of test window. If None, same as window_size
        gap : int
            Number of samples to skip between train and test (default: 0)
        """
        self.n_splits = n_splits
        self.window_size = window_size
        self.test_size = test_size
        self.gap = gap
    
    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits using sliding window approach.
        
        Yields
        ------
        train_idx : np.ndarray
            Indices for training set
        test_idx : np.ndarray
            Indices for validation/test set
        """
        n_samples = len(X)
        
        # Calculate window sizes if not provided
        if self.window_size is None:
            # Default: divide data into n_splits + 1 equal parts
            self.window_size = n_samples // (self.n_splits + 1)
        
        if self.test_size is None:
            self.test_size = self.window_size
        
        # Calculate valid number of splits based on available data
        # We need: window_size * 1 + test_size * n_splits <= n_samples
        max_splits = (n_samples - self.window_size) // self.test_size
        actual_splits = min(self.n_splits, max_splits)
        
        if actual_splits < 1:
            raise ValueError(
                f"Not enough data for {self.n_splits} splits. "
                f"Need at least {self.window_size + self.test_size} samples, got {n_samples}"
            )
        
        for i in range(actual_splits):
            # Training window: [start, start + window_size)
            train_start = i * self.test_size
            train_end = train_start + self.window_size
            
            # Test window: [train_end + gap, train_end + gap + test_size)
            test_start = train_end + self.gap
            test_end = test_start + self.test_size
            
            # Ensure we don't go beyond data
            if test_end > n_samples:
                break
            
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            
            yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations."""
        return self.n_splits


def create_sliding_window_splitter(
    n_splits: int,
    train_window_ratio: float = 0.5,
    test_window_ratio: float = 0.5,
    gap: int = 0
):
    """
    Factory function to create a SlidingWindowSplit with ratio-based window sizes.
    
    Parameters
    ----------
    n_splits : int
        Number of folds
    train_window_ratio : float
        Ratio of data used for training window (e.g., 0.5 = 50% for train)
    test_window_ratio : float
        Ratio of data used for test window
    gap : int
        Gap between train and test
    
    Returns
    -------
    SlidingWindowSplit
    """
    # Store ratios and create splitter later when data size is known
    class RatioSlidingWindowSplit(SlidingWindowSplit):
        def __init__(self, n_splits, train_ratio, test_ratio, gap):
            super().__init__(n_splits=n_splits, gap=gap)
            self.train_ratio = train_ratio
            self.test_ratio = test_ratio
        
        def split(self, X, y=None, groups=None):
            n_samples = len(X)
            # Calculate window sizes based on ratios of total data
            # Each fold uses train_ratio + test_ratio of data
            self.window_size = int(n_samples * self.train_ratio / self.n_splits)
            self.test_size = int(n_samples * self.test_ratio / self.n_splits)
            return super().split(X, y, groups)
    
    return RatioSlidingWindowSplit(n_splits, train_window_ratio, test_window_ratio, gap)
