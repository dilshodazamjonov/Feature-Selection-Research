import numpy as np
from typing import Iterator, Tuple


def _to_1d_proba(values):
    """Convert 2D probability array to 1D, taking the positive class column."""
    values = np.asarray(values)
    if values.ndim == 2:
        if values.shape[1] == 1:
            values = values[:, 0]
        else:
            values = values[:, 1]
    return values.reshape(-1)


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
