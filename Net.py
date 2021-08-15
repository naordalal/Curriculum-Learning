from torch import nn
import random
import torch
from utils import *

class Net(nn.Module):
    def __init__(self, in_channels, n_classes, loss_fn, optimizer, device, kind, pacing_func=None, batch_size=100, epochs=150,
                 starting_percent=0.1, increase_amount=2, step_length=250, lr_step_length=200, initial_lr=0.1, decay_lr=2):
        super().__init__()
        self.kind = kind
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.device = device
        self.pacing_func = pacing_func
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.epochs = epochs
        self.starting_percent = starting_percent
        self.increase_amount = increase_amount
        self.step_length = step_length
        self.lr_step_length = lr_step_length
        self.initial_lr = initial_lr
        self.decay_lr = decay_lr
        self.epochs = epochs

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ELU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ELU(0.2),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ELU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ELU(0.2),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ELU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ELU(0.2),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, kernel_size=2, padding='same'),
            nn.BatchNorm2d(256),
            nn.ELU(0.2),
            nn.Conv2d(256, 256, kernel_size=2, padding='same'),
            nn.BatchNorm2d(256),
            nn.ELU(0.2),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ELU(0.2),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes),
            nn.Softmax(dim=1),
            ).to(device)
        self.optimizer = optimizer(self.parameters(), initial_lr)
        if 'linear' in kind:
            self.pacing_params = [self.starting_percent, self.increase_amount]
        elif 'curriculum' in kind:
            self.pacing_params = [self.starting_percent, self.increase_amount, self.step_length]

        
    def forward(self, x):
        return self.network(x)
        
    def build_dataloader(self, x_, y_):
        tensor_x = torch.Tensor(x_)
        tensor_y = torch.Tensor(y_)
        my_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        return torch.utils.data.DataLoader(my_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False)

    def _get_random_batch(self, data_loader):
        subset_indices = np.random.choice(len(data_loader), self.batch_size, replace=False)
        subset = torch.utils.data.Subset(data_loader.dataset, subset_indices)
        return torch.utils.data.DataLoader(subset, batch_size=1, num_workers=0, shuffle=False)
        return list(subset)

    def _score(self, validation_dataloader):
        valid_actual = []
        for batch in validation_dataloader:
            valid_actual += list(batch[1].numpy())
        valid_predicted = self.predict_proba(validation_dataloader)
        valid_predicted = np.argmax(valid_predicted, axis=1)
        return sum(np.array(valid_actual) == np.array(valid_predicted)) / len(valid_actual)

    def predict_proba(self, validation_dataloader):
        self.eval()
        valid_predicted = []

        for batch in validation_dataloader:
            batch_x = batch[0].to(self.device)
            outputs = self(batch_x).squeeze()
            valid_predicted.append(outputs.detach().cpu().numpy())
        return np.concatenate(valid_predicted, axis = 0)

    def fit(self, train_x, train_y, val_x, val_y, validation_every = 20):
        valid_acc = []
        self.train()
        n_batches = self.epochs * len(train_y) // self.batch_size
        validation_dl = self.build_dataloader(val_x, val_y)
        train_x, train_y = organize_data(train_x, train_y, self.kind, self.device, self.batch_size)
        
        if 'curriculum' == self.kind:
            i = 0
            pacing = self.pacing_func(*self.pacing_params)
            done_training = False
            while not done_training:
                curr_x, curr_y = pacing(train_x, train_y, i)
                dl = self.build_dataloader(curr_x, curr_y)
                g = iter(dl)

                n_ = self.step_length if len(curr_y) < len(train_y) else n_batches - i
                for j in range(n_):
                    try:
                      batch = next(g)
                    except StopIteration:
                      g = iter(dl)
                      batch = next(g)
                    batch_x = batch[0].to(self.device)
                    batch_y = batch[1].to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self(batch_x)
                    if len(batch_x) != 1:
                        outputs = outputs.squeeze()
                        
                    loss = self.loss_fn(outputs, batch_y.long())
                    loss.backward()
                    self.optimizer.step()
                    if i % validation_every == 0:
                        valid_acc.append(self._score(validation_dl))
                        self.train()
                    i += 1
                    if i == n_batches:
                        done_training = True
                        break
                    
        elif self.kind == 'linear_curriculum':
            pacing = self.pacing_func(*self.pacing_params)
            for i in range(n_batches):
                curr_x, curr_y = pacing(train_x, train_y, i)
                indexes = np.random.choice(len(curr_y), min(100, len(curr_y)), replace=False)
                batch_x = torch.tensor(curr_x[indexes]).to(self.device)
                batch_y = torch.tensor(curr_y[indexes]).to(self.device)
                self.optimizer.zero_grad()
                outputs = self(batch_x)
                if len(batch_x) != 1:
                    outputs = outputs.squeeze()
                loss = self.loss_fn(outputs, batch_y.long())
                loss.backward()
                self.optimizer.step()
                if i % validation_every == 0:
                    valid_acc.append(self._score(validation_dl))
                    self.train()
        else:
            train_dl = self.build_dataloader(train_x, train_y)
            g = iter(train_dl)
            for i in range(n_batches):
                try:
                    batch = next(g)
                except StopIteration:
                    g = iter(train_dl)
                    batch = next(g)
                batch_x = batch[0].to(self.device)
                batch_y = batch[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self(batch_x)
                if len(batch_x) != 1:
                    outputs = outputs.squeeze()
                loss = self.loss_fn(outputs, batch_y.long())
                loss.backward()
                self.optimizer.step()
                if i % validation_every == 0:
                    valid_acc.append(self._score(validation_dl))
                    self.train()
        return self._score(validation_dl), valid_acc