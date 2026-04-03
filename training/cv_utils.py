import numpy as np

def _instantiate(cls_or_obj, kwargs):
    """
    Instantiates a class with provided keyword arguments or returns the object if already instantiated.
    """
    return cls_or_obj(**kwargs) if isinstance(cls_or_obj, type) else cls_or_obj


def _fit_selector(selector, X_train, y_train):
    """
    Fits the feature selector to the training data and returns the transformed features,
    handling various scikit-learn API signatures.
    """
    try:
        return selector.fit_transform(X_train, y_train)
    except TypeError:
        try:
            return selector.fit_transform(X_train)
        except TypeError:
            selector.fit(X_train, y_train)
            return selector.transform(X_train)


def _transform_selector(selector, X):
    """
    Applies the transformation of a previously fitted selector to the input data.
    """
    return selector.transform(X)



def _to_1d_proba(values):
    values = np.asarray(values)
    if values.ndim == 2:
        if values.shape[1] == 1:
            values = values[:, 0]
        else:
            values = values[:, 1]
    return values.reshape(-1)
