import torch as t


def macro_soft_f1(real_vals, predictions, reduction):
    """from https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d"""
    true_positive = (real_vals * predictions).sum(dim=0)
    false_positive = (predictions * (1 - real_vals)).sum(dim=0)
    false_negative = ((1 - predictions) * real_vals).sum(dim=0)
    soft_f1 = 2 * true_positive / (2 * true_positive + false_negative + false_positive + 1e-16)
    if reduction == "mean":
        loss = t.mean(1 - soft_f1)
    else:
        loss = 1 - soft_f1
    return loss


def coral_loss(logits, levels, importance_weights=None, reduction="mean"):
    """Computes the CORAL loss described in
    Cao, Mirjalili, and Raschka (2020)
    *Rank Consistent Ordinal Regression for Neural Networks
       with Application to Age Estimation*
    Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008
    Parameters
    ----------
    logits : torch.tensor, shape(num_examples, num_classes-1)
        Outputs of the CORAL layer.
    levels : torch.tensor, shape(num_examples, num_classes-1)
        True labels represented as extended binary vectors
        (via `coral_pytorch.dataset.levels_from_labelbatch`).
    importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
        Optional weights for the different labels in levels.
        A tensor of ones, i.e.,
        `torch.ones(num_classes-1, dtype=torch.float32)`
        will result in uniform weights that have the same effect as None.
    reduction : str or None (default='mean')
        If 'mean' or 'sum', returns the averaged or summed loss value across
        all data points (rows) in logits. If None, returns a vector of
        shape (num_examples,)
    Returns
    ----------
        loss : torch.tensor
        A torch.tensor containing a single loss value (if `reduction='mean'` or '`sum'`)
        or a loss value for each data record (if `reduction=None`).
    Examples
    ----------
    >>> import torch
    >>> from coral_pytorch.losses import coral_loss
    >>> levels = torch.tensor(
    ...    [[1., 1., 0., 0.],
    ...     [1., 0., 0., 0.],
    ...    [1., 1., 1., 1.]])
    >>> logits = torch.tensor(
    ...    [[2.1, 1.8, -2.1, -1.8],
    ...     [1.9, -1., -1.5, -1.3],
    ...     [1.9, 1.8, 1.7, 1.6]])
    >>> coral_loss(logits, levels)
    tensor(0.6920)
    https://github.com/Raschka-research-group/coral-pytorch/blob/c6ab93afd555a6eac708c95ae1feafa15f91c5aa/coral_pytorch/losses.py
    """

    if not logits.shape == levels.shape:
        raise ValueError(
            "Please ensure that logits (%s) has the same shape as levels (%s). " % (logits.shape, levels.shape)
        )

    term1 = t.nn.functional.logsigmoid(logits) * levels + (t.nn.functional.logsigmoid(logits) - logits) * (1 - levels)

    if importance_weights is not None:
        term1 *= importance_weights

    val = -t.sum(term1, dim=1)

    if reduction == "mean":
        loss = t.mean(val)
    elif reduction == "sum":
        loss = t.sum(val)
    elif reduction is None:
        loss = val
    else:
        s = 'Invalid value for `reduction`. Should be "mean", ' '"sum", or None. Got %s' % reduction
        raise ValueError(s)

    return loss


def corn_loss(logits, y_train, num_classes):
    """Computes the CORN loss described in our forthcoming
    'Deep Neural Networks for Rank Consistent Ordinal
    Regression based on Conditional Probabilities'
    manuscript.
    Parameters
    ----------
    logits : torch.tensor, shape=(num_examples, num_classes-1)
        Outputs of the CORN layer.
    y_train : torch.tensor, shape=(num_examples)
        Torch tensor containing the class labels.
    num_classes : int
        Number of unique class labels (class labels should start at 0).
    Returns
    ----------
        loss : torch.tensor
        A torch.tensor containing a single loss value.
    Examples
    ----------
    >>> import torch
    >>> from coral_pytorch.losses import corn_loss
    >>> # Consider 8 training examples
    >>> _  = torch.manual_seed(123)
    >>> X_train = torch.rand(8, 99)
    >>> y_train = torch.tensor([0, 1, 2, 2, 2, 3, 4, 4])
    >>> NUM_CLASSES = 5
    >>> #
    >>> #
    >>> # def __init__(self):
    >>> corn_net = torch.nn.Linear(99, NUM_CLASSES-1)
    >>> #
    >>> #
    >>> # def forward(self, X_train):
    >>> logits = corn_net(X_train)
    >>> logits.shape
    torch.Size([8, 4])
    >>> corn_loss(logits, y_train, NUM_CLASSES)
    tensor(0.7127, grad_fn=<DivBackward0>)
    https://github.com/Raschka-research-group/coral-pytorch/blob/c6ab93afd555a6eac708c95ae1feafa15f91c5aa/coral_pytorch/losses.py
    """
    sets = []
    for i in range(num_classes - 1):
        label_mask = y_train > i - 1
        label_tensor = (y_train[label_mask] > i).to(t.int64)
        sets.append((label_mask, label_tensor))

    num_examples = 0
    losses = 0.0
    for task_index, s in enumerate(sets):
        train_examples = s[0]
        train_labels = s[1]

        if len(train_labels) < 1:
            continue

        num_examples += len(train_labels)
        pred = logits[train_examples, task_index]

        loss = -t.sum(
            t.nn.functional.logsigmoid(pred) * train_labels
            + (t.nn.functional.logsigmoid(pred) - pred) * (1 - train_labels)
        )
        losses += loss

    return losses / num_examples


def corn_label_from_logits(logits):
    """
    Returns the predicted rank label from logits for a
    network trained via the CORN loss.
    Parameters
    ----------
    logits : torch.tensor, shape=(n_examples, n_classes)
        Torch tensor consisting of logits returned by the
        neural net.
    Returns
    ----------
    labels : torch.tensor, shape=(n_examples)
        Integer tensor containing the predicted rank (class) labels
    Examples
    ----------
    >>> # 2 training examples, 5 classes
    >>> logits = torch.tensor([[14.152, -6.1942, 0.47710, 0.96850],
    ...                        [65.667, 0.303, 11.500, -4.524]])
    >>> corn_label_from_logits(logits)
    tensor([1, 3])
    https://github.com/Raschka-research-group/coral-pytorch/blob/c6ab93afd555a6eac708c95ae1feafa15f91c5aa/coral_pytorch/dataset.py
    """
    probas = t.sigmoid(logits)
    probas = t.cumprod(probas, dim=1)
    predict_levels = probas > 0.5
    predicted_labels = t.sum(predict_levels, dim=1)
    return predicted_labels
