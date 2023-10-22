# This code snippet aims to simulate a bigger batch size when it cannot be
# achieved due to limited GPU memory.
#
# Chen Liu (chen.liu.cl2482@yale.edu)

import argparse
import torch


class FakeDataest(torch.utils.data.Dataset):

    def __init__(self,
                 num_samples: int = 1000,
                 num_classes: int = 10,
                 random_seed: int = 0):
        super().__init__()
        self.num_samples = num_samples
        self.num_classes = num_classes
        torch.manual_seed(random_seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, _idx) -> torch.Tensor:
        x = torch.randn((3, 64, 64))
        y = torch.randint(low=0, high=self.num_classes, size=(1, ))[0]
        return x, y


class SmallConvNet(torch.nn.Module):

    def __init__(self,
                 num_classes: int = 10,
                 image_shape: str = (3, 64, 64)) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Get the correct dimensions of the classifer.
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=5), torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.Conv2d(64, 64,
                            kernel_size=5), torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.Conv2d(64, 64, kernel_size=5),
            torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(num_features=64),
            torch.nn.Flatten())

        sample_input = torch.ones((1, *image_shape))
        sample_output = self.encoder(sample_input)
        assert len(sample_output.shape) == 2

        self.linear = torch.nn.Linear(in_features=sample_output.shape[-1],
                                      out_features=self.num_classes)

    def forward(self, x):
        return self.linear(self.encoder(x))


def train(args):
    device = torch.device(
        'cuda:%d' % args['gpu_id'] if torch.cuda.is_available() else 'cpu')

    B_actual = args['batch_size_actual']
    B_desired = args['batch_size_desired']

    if B_desired % B_actual == 0:
        grad_update_freq = B_desired // B_actual
    else:
        raise ValueError(
            'Currently do not support `batch_size_desired` not divisible by `batch_size_actual`.'
        )

    train_dataset = FakeDataest()
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=B_actual,
                                               shuffle=True,
                                               num_workers=args['num_workers'],
                                               pin_memory=True)

    model = SmallConvNet()
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(list(model.parameters()),
                            lr=float(args['learning_rate']))
    '''

    Pay attention to the gradient backprop.

    '''
    for _ in range(args['train_epochs']):
        opt.zero_grad()
        for batch_idx, (x, y_true) in enumerate(train_loader):
            x, y_true = x.to(device), y_true.to(device)

            y_pred = model(x)

            # NOTE: We need to reduce the loss by a factor of `grad_update_freq`,
            # assuming the loss uses mean aggregation (which is the case for many loss fns).
            loss = loss_fn(y_pred, y_true) / grad_update_freq
            '''

            Usually we would do:

            opt.zero_grad()
            loss.backward()
            opt.step()

            But here we will need to simulate a bigger batch size
            by accumulating the gradients.

            '''

            loss.backward()

            shall_backprop = batch_idx % grad_update_freq == (
                grad_update_freq - 1)
            if shall_backprop:
                opt.step()
                opt.zero_grad()

            # Printing to check that the model weights are indeed
            # only updated when a `batch_size_desired` is reached.
            print(
                'Are we backproping this batch? %s. Sample model weight: %s' %
                (shall_backprop,
                 str(model.encoder[6].weight[0][0][0].cpu().detach().numpy())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point.')
    parser.add_argument('--gpu-id', help='Index of GPU device', default=0)
    parser.add_argument('--train-epochs',
                        help='Number of training epochs',
                        default=10)
    parser.add_argument('--learning-rate', default=1e-3)
    parser.add_argument('--num-workers',
                        help='Number of workers for data loading',
                        default=1)
    parser.add_argument('--batch-size-actual',
                        help='Actual batch size (limited by GPU)',
                        default=2)
    parser.add_argument('--batch-size-desired',
                        help='Desired batch size',
                        default=10)
    args = vars(parser.parse_args())

    train(args)
