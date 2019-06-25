
def standardise(X, how='range=2', mean=None, std=None, midrange=None,
                ptp=None):
    """Standardise.
    ftp://ftp.sas.com/pub/neural/FAQ2.html#A_std_in
    Parameters
    ----------
    X : matrix
        Each sample is in range [0, 1]
    how : str, {'range=2', 'std=1'}
        'range=2' sets midrange to 0 and enforces
        all values to be in the range [-1,1]
        'std=1' sets mean = 0 and std = 1
    Returns
    -------
    new_X : matrix
        Same shape as `X`.  Sample is in range [lower, upper]
    See also
    --------
    unstandardise
    """
    if how == 'std=1':
        if mean is None:
            mean = X.mean()
        if std is None:
            std = X.std()
        centered = X - mean
        if std == 0:
            return centered
        else:
            return centered / std
    elif how == 'range=2':
        if midrange is None:
            midrange = (X.max() + X.min()) / 2
        if ptp is None:
            ptp = X.ptp()
        return (X - midrange) / (ptp / 2)
    else:
        raise RuntimeError("unrecognised how '" + how + "'")

def mains_to_batches(mains, n_seq_per_batch=128, seq_length=100, std=None, stride=1):
    """
    Parameters
    ----------
    mains : 1D np.ndarray Watts.
        And it is highly advisable to pad `mains` with `seq_length` elements
        at both ends so the net can slide over the very start and end.
    std : mains standard deviation
    stride : int, optional
    Returns
    -------
    batches : list of 3D arrays
    """
    assert mains.ndim == 1
    n_mains_samples = len(mains)
    input_shape = (n_seq_per_batch, seq_length, 1)

    # Divide mains data into batches
    n_batches = (n_mains_samples / stride) / n_seq_per_batch
    n_batches = np.ceil(n_batches).astype(int)
    batches = []
    for batch_i in range(n_batches):
        batch = np.zeros(input_shape, dtype=np.float32)
        batch_start = batch_i * n_seq_per_batch * stride
        for seq_i in range(n_seq_per_batch):
            mains_start_i = batch_start + (seq_i * stride)
            mains_end_i = mains_start_i + seq_length
            seq = mains[mains_start_i:mains_end_i]
            #seq_standardised = normalise(seq)
            seq_standardised = seq/(4621.69)
            #seq_standardised = standardise(seq, how='std=1', std=std)
            batch[seq_i, :len(seq), 0] = seq_standardised
        batches.append(batch)
        #yield batches

    return batches
  
  
def disag_ae_or_rnn(mains, net, max_target_power,std=None, stride=1):
    """
    Parameters
    ----------
    mains : 1D np.ndarray
        Watts.
        Mains must be padded with at least `seq_length` elements
        at both ends so the net can slide over the very start and end.
    net : neuralnilm.net.Net
    max_target_power : int
        Watts
    stride : int or None, optional
        if None then stide = seq_length
    Returns
    -------
    estimates : 1D vector
    """
    #n_seq_per_batch, seq_length = net.input_shape[:2]
    n_seq_per_batch = 128
    seq_length = 100
    #model = load_model(filepath)
    if stride is None:
        stride = seq_length
    batches = mains_to_batches(mains, n_seq_per_batch, seq_length, std, stride)
    #print(batches)
    estimates = np.zeros(len(mains), dtype=np.float32)
    assert not seq_length % stride

    # Iterate over each batch
    for batch_i, net_input in enumerate(batches):
        #print(net_input)
        net_output = net.predict(net_input)
        #print(net_output)
        batch_start = batch_i * n_seq_per_batch * stride
        for seq_i in range(n_seq_per_batch):
            start_i = batch_start + (seq_i * stride)
            end_i = start_i + seq_length
            n = len(estimates[start_i:end_i])
            # The net output is not necessarily the same length
            # as the mains (because mains might not fit exactly into
            # the number of batches required)
            estimates[start_i:end_i] += net_output[seq_i, :n]

    n_overlaps = seq_length / stride
    estimates /= n_overlaps
    #estimates *= max_target_power
    estimates *=4621.69
    estimates[estimates < 0] = 0
    return estimates