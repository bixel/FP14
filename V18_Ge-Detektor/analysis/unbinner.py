def unbinned_array(binned_array, start=0, end=None,
        calibration_func=lambda x: x):
    """ Returned an array with 'raw' Data for creating histograms with plt.hist
    """
    raw_data = []
    if end is None:
        end = len(binned_array) - 1
    for x in range(start, end):
        raw_data.extend([calibration_func(x)] * binned_array[x])
    return raw_data
