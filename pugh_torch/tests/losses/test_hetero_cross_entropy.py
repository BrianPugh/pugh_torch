import torch
from torch import nn
import torch.nn.functional as F
from pugh_torch.losses import hetero_cross_entropy
from pugh_torch.losses.hetero_cross_entropy import _format_weight
import pytest
import numpy as np


def test_format_weight_numpy():
    weight = np.ones((3, 3))
    actual = _format_weight(weight, None, "cpu")


def test_format_weight_tensor():
    pass


def test_format_weight_scalar():
    pass


def test_format_weight_error():
    with pytest.raises(ValueError):
        _format_weight("foo", (1, 2), "cpu")


@pytest.fixture
def pred():
    """
    (1, 4, 2, 3) logits
    """

    data = torch.FloatTensor(
        [
            [  # Batch 0
                [
                    [1.1, 0.5, 0.2],
                    [1.3, 0.1, 0.6],
                ],  # Class 0
                [
                    [0.7, 0.4, 1.5],
                    [6, 0.1, -2],
                ],  # Class 1
                [
                    [1.7, 0.25, -3],
                    [4, 0.9, 0.8],
                ],  # Class 2
                [
                    [1, 2, 3],
                    [3, 2, 1],
                ],  # Class 3
            ]
        ],
    )
    data.requires_grad_()
    return data


def test_hetero_cross_entropy_ce_only(pred):
    """Should behave as normal cross entropy when no superclass index is
    specified.
    """
    # (1,2,3)
    target = torch.LongTensor(
        [
            [
                [-2, 1, -2],
                [-2, -2, 1],
            ]
        ]
    )  # Height
    available = torch.BoolTensor(
        [
            [True, True, True, True],
        ]
    )

    actual_loss = hetero_cross_entropy(pred, target, available, ignore_index=-2)
    actual_loss.backward()  # This should always work

    actual_loss = actual_loss.detach().numpy()

    # Compute what the expected value should be
    pred_valid = torch.FloatTensor([[[0.5, 0.4, 0.25, 2], [0.6, -2, 0.8, 1],]]).permute(
        0, 2, 1
    )  # (1, 4, 2)
    target_valid = torch.LongTensor([[1, 1]])
    expected = F.cross_entropy(pred_valid, target_valid)  # This should be 3.0005

    assert np.isclose(actual_loss, expected.detach().numpy())


def test_hetero_cross_entropy_super_only_simple():
    pred = torch.Tensor([-1, 0, 1, 2]).reshape(1, 4, 1)  # logits
    target = torch.full((1, 1), -1, dtype=torch.long)
    available = torch.BoolTensor([True, True, False, False]).reshape(1, -1)

    actual_loss = hetero_cross_entropy(
        pred, target, available, ignore_index=-2, super_index=-1
    )
    actual_loss.backward()
    actual_loss = actual_loss.detach().numpy()

    assert np.isclose(0.12692809, actual_loss)


def test_hetero_cross_entropy_super_only(pred):
    target = torch.LongTensor(
        [
            [
                [-1, -1, -1],
                [-1, -1, -1],
            ]
        ]
    )  # Height
    available = torch.BoolTensor(
        [
            [True, False, False, False],
        ]  # Only class 0 is available.
    )

    actual_loss = hetero_cross_entropy(
        pred, target, available, ignore_index=-2, super_index=-1
    )
    actual_loss.backward()  # This should always work

    actual_loss = actual_loss.detach().numpy()
    assert np.isclose(actual_loss, 0.14451277)


def test_hetero_cross_entropy_all_invalid(pred):
    """This primarily tests that when all invalid data is provided (i.e.
    the returned loss is 0) that the ``backward()`` method still works.
    """

    target = torch.LongTensor(
        [
            [
                [-2, -2, -2],
                [-2, -2, -2],
            ]
        ]
    )  # Height
    available = torch.BoolTensor(
        [
            [True, True, True, True],
        ]
    )

    actual_loss = hetero_cross_entropy(
        pred, target, available, ignore_index=-2, super_index=-1
    )
    actual_loss.backward()  # This should always work

    actual_loss = actual_loss.detach().numpy()
    assert actual_loss == 0


def test_hetero_cross_entropy_complete(pred):
    """Test both parts (ce_loss + super_loss) combined"""

    target = torch.LongTensor(
        [
            [
                [-1, 1, -2],
                [-2, -2, 1],
            ]
        ]
    )  # Height
    available = torch.BoolTensor(
        [
            [True, True, False, False],
        ]
    )

    pred_softmax = F.softmax(pred, dim=1)  # For inspecting/debugging purposes
    """
    Predicted classes (where X means doesn't matter AT ALL):
    tensor([[[2, 3, x],
             [x, x, 3]]])

    [0,0] -> 2 is the interesting one here.
        If the loss is working, this prediction should increase 2/3 class prob
        # pred - learning_rate * grad  should get us closer.
    """

    actual_loss = hetero_cross_entropy(
        pred, target, available, ignore_index=-2, super_index=-1
    )
    actual_loss.backward()  # This should always work

    pred_grad = pred.grad.detach().numpy()
    assert not pred_grad[:, :, 0, 2].any()
    assert not pred_grad[:, :, 1, 0].any()
    assert not pred_grad[:, :, 1, 1].any()

    actual_loss = actual_loss.detach().numpy()
    assert np.isclose(actual_loss, 3.4782794)

    updated_pred = pred - (0.1 * pred.grad)
    updated_pred_softmax = F.softmax(updated_pred, dim=1)

    assert (
        updated_pred_softmax[0, 1, 0, 1] > pred_softmax[0, 1, 0, 1]
    )  # prediction should be more sure of 1 target
    assert (
        updated_pred_softmax[0, 1, 1, 2] > pred_softmax[0, 1, 1, 2]
    )  # prediction should be more sure of 1 target

    # The classes that are in the dataset should now have lower probabilities
    # for the pixel that's marked as unlabeled.
    assert updated_pred_softmax[0, 0, 0, 0] < pred_softmax[0, 0, 0, 0]
    assert updated_pred_softmax[0, 1, 0, 0] < pred_softmax[0, 1, 0, 0]


def test_hetero_cross_entropy_super_only_simple_alpha_near_zero():
    pred = torch.Tensor([-1, 0, 1, 2]).reshape(1, 4, 1)  # logits
    target = torch.full((1, 1), -1, dtype=torch.long)
    available = torch.BoolTensor([True, True, False, False]).reshape(1, -1)

    actual_loss = hetero_cross_entropy(
        pred, target, available, ignore_index=-2, super_index=-1, alpha=0.0001
    )
    actual_loss.backward()
    actual_loss = actual_loss.detach().numpy()

    assert np.isclose(0.12692809, actual_loss, rtol=0.01)


def test_hetero_cross_entropy_smoothing_complete_alpha_near_zero(pred):
    """Test both parts (ce_loss + super_loss) combined + label smoothing"""

    target = torch.LongTensor(
        [
            [
                [-1, 1, -2],
                [-2, -2, 1],
            ]
        ]
    )  # Height
    available = torch.BoolTensor(
        [
            [True, True, False, False],
        ]
    )

    pred_softmax = F.softmax(pred, dim=1)  # For inspecting/debugging purposes
    """
    Predicted classes (where X means doesn't matter AT ALL):
    tensor([[[2, 3, x],
             [x, x, 3]]])

    [0,0] -> 2 is the interesting one here.
        If the loss is working, this prediction should increase 2/3 class prob
        # pred - learning_rate * grad  should get us closer.
    """

    actual_loss = hetero_cross_entropy(
        pred,
        target,
        available,
        ignore_index=-2,
        super_index=-1,
        alpha=0.00001,
    )

    actual_loss.backward()  # This should always work

    pred_grad = pred.grad.detach().numpy()
    assert not pred_grad[:, :, 0, 2].any()
    assert not pred_grad[:, :, 1, 0].any()
    assert not pred_grad[:, :, 1, 1].any()

    actual_loss = actual_loss.detach().numpy()
    assert np.isclose(actual_loss, 3.4782794, rtol=0.01)

    updated_pred = pred - (0.1 * pred.grad)
    updated_pred_softmax = F.softmax(updated_pred, dim=1)

    assert (
        updated_pred_softmax[0, 1, 0, 1] > pred_softmax[0, 1, 0, 1]
    )  # prediction should be more sure of 1 target
    assert (
        updated_pred_softmax[0, 1, 1, 2] > pred_softmax[0, 1, 1, 2]
    )  # prediction should be more sure of 1 target

    # The classes that are in the dataset should now have lower probabilities
    # for the pixel that's marked as unlabeled.
    assert updated_pred_softmax[0, 0, 0, 0] < pred_softmax[0, 0, 0, 0]
    assert updated_pred_softmax[0, 1, 0, 0] < pred_softmax[0, 1, 0, 0]


def test_hetero_cross_entropy_weight_int(pred):
    """Test both parts (ce_loss + super_loss) combined plus weight"""

    target = torch.LongTensor(
        [
            [
                [-1, 1, -2],
                [-2, -2, 1],
            ]
        ]
    )
    available = torch.BoolTensor(
        [
            [True, True, False, False],
        ]
    )

    actual_loss = hetero_cross_entropy(
        pred, target, available, ignore_index=-2, super_index=-1, weight=2
    )
    actual_loss.backward()  # This should always work

    actual_loss = actual_loss.detach().numpy()
    assert np.isclose(actual_loss, 2 * 3.4782794)


def test_hetero_cross_entropy_weight_tensor_no_class(pred):
    """Test both parts (ce_loss + super_loss) combined plus weight"""

    target = torch.LongTensor(
        [
            [
                [-1, 1, -2],
                [-2, -2, 1],
            ]
        ]
    )
    available = torch.BoolTensor(
        [
            [True, True, False, False],
        ]
    )
    weight = torch.Tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    ).reshape(1, 1, 2, 3)

    actual_loss = hetero_cross_entropy(
        pred, target, available, ignore_index=-2, super_index=-1, weight=weight
    )
    actual_loss.backward()  # This should always work

    actual_loss = actual_loss.detach().numpy()
    assert np.isclose(actual_loss, 14.342173)

class DigitRecognizerCNN(nn.Module):
    """Simple convolutional network consisting of 2 convolution layers with
    max-pooling followed by two fully-connected layers and a softmax output
    layer

    https://www.kaggle.com/mcwitt/mnist-cnn-with-pytorch
    """

    def __init__(self, num_classes):

        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(32 * 7 * 7, 784),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(784, 784),
            nn.ReLU(),
            nn.Linear(784, num_classes))

    def forward(self, X):
        x = self.features(X)
        x = x.view(x.size(0), 32 * 7 * 7)
        x = self.classifier(x)
        return x

@pytest.mark.network
def test_hetero_cross_entropy_mnist():
    """ Runs a small network on mnist to observe this loss on a toy problem/network.

    Treat this as an integration test.
    """

    from pugh_torch.datasets.classification import MNIST
    from torchvision import transforms
    import matplotlib.pyplot as plt

    # Seed RNG for determinism
    def make_deterministic():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)

    transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
    )

    batch_size = 32

    trainset = MNIST(split="train", transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    valset = MNIST(split="val", transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, num_workers=2)

    print_interval = 2000 // batch_size 
    split_thresh = 3
    def run_train(model, naive=False, hetero=False):
        for epoch in range(1):
            running_loss = 0.0  # Accumulates loss over print_interval
            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                optimizer.zero_grad()

                availables = torch.ones(batch_size, 10).to('cuda')

                if i % 2 == 0:
                    mask = labels < split_thresh  # 0, 1, and 2
                    if naive:
                        # Equivalent to only apply loss for labeled data
                        labels = labels[mask]
                        inputs = inputs[mask]
                    elif hetero:
                        labels[~mask] = -1
                        availables[~mask, split_thresh:] = False

                outputs = net(inputs)

                if hetero:
                    loss = hetero_cross_entropy(outputs, labels, availables, super_index=-1)
                else:
                    loss = F.cross_entropy(outputs, labels)

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if True and i % print_interval == (print_interval-1):
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / print_interval))
                    running_loss = 0.0


    def run_val(model):
        correct = np.zeros(10)
        total = np.zeros(10)
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                outputs = model(inputs)
                _, predicted = torch.max(outputs, -1)

                predicted = predicted.cpu().numpy()
                labels = labels.cpu().numpy()
                for predict, label in zip(predicted, labels):
                    correct[label] += (predict==label)
                    total[label] += 1

        accuracy = correct.sum() / total.sum()
        print('Accuracy of the network on the 10000 test images: %.2f %%' % (
            100 * accuracy))
        for i, (c, t) in enumerate(zip(correct, total)):
            print(f'    label {i}: {100*c/t:.2f}')
        return correct / total


    expected = np.array([0.99285714, 0.99207048, 0.98643411, 0.98019802, 0.97046843,
               0.98318386, 0.95302714, 0.98735409, 0.97741273, 0.90882061])
    if False:
        make_deterministic()
        net = DigitRecognizerCNN(10).to('cuda')
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
        print("Training classic cross-entropy network")
        run_train(net)
        print('Finished Training. Running validation split...')
        acc = run_val(net)

        assert np.allclose(acc, expected, atol=0.0001)

    # Naive Cross-Entropy with non-heterogeneous datasets
    # Essentially, only apply cross-entropy loss for labeled datapoints.
    expected_naive =  np.array([0.99795918, 0.99471366, 0.98837209, 0.96831683, 0.9592668 ,
               0.94170404, 0.95093946, 0.96595331, 0.94661191, 0.95936571])
    if False:
        make_deterministic()
        net = DigitRecognizerCNN(10).to('cuda')
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
        print("Training naive cross-entropy network")
        run_train(net, naive=True)
        print('Finished Training. Running validation split...')
        acc_naive = run_val(net)
        assert np.allclose(acc_naive, expected_naive, atol=0.0001)

    # Hetero-Cross-Entropy
    expected_hetero = np.array([0.99081633, 0.99559471, 0.98546512, 0.98415842, 0.97861507,
               0.97197309, 0.97807933, 0.97568093, 0.9825462 , 0.9444995 ])
    if True:
        make_deterministic()
        net = DigitRecognizerCNN(10).to('cuda')
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
        print("Training hetero-cross-entropy network")
        run_train(net, hetero=True)
        print('Finished Training. Running validation split...')
        acc_hetero = run_val(net)
        assert np.allclose(acc_hetero, expected_hetero, atol=0.0001)

