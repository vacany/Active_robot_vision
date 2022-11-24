import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import torch

x, y = make_classification(n_samples=100, n_features=2,
                           n_informative=2, n_redundant=0, n_repeated=0)
x = (x - x.min()) / (x.max() - x.min())

class Active_dataset():
    def __init__(self, x, y, init_size=10):
        x = torch.tensor(x, dtype=torch.float, requires_grad=True)
        y = torch.tensor(y, dtype=torch.long)
        self.init_size = init_size

        self.active_train = x[:init_size]
        self.active_label = y[:init_size]

        self.x_pool = x[init_size:]
        self.y_pool = y[init_size:]

    def get_pool_sample(self, idx):
        return self.x_pool[idx], self.y_pool[idx]

    def get_active_data(self):
        return self.active_train, self.active_label

    def get_whole_data(self):
        return self.x, self.y

    def update_active(self, x, y):
        self.active_train = torch.cat((self.active_train, x[None,:]))
        self.active_label = torch.cat((self.active_label, y[None,:]))

    def calculate_active_accuracy(self, y_hat):
        acc = np.mean(self.active_label == y_hat)
        return acc

def draw(x, y, y_hat):
    fig, axes = plt.subplots(1,2)
    axes[0].scatter(*x.T.detach(), c=y.detach())
    axes[1].scatter(*x.T.detach(), c=torch.argmax(y_hat, dim=1).detach())

    axes[0].set_title('Ground Truth')
    axes[1].set_title('Prediction')
    plt.show()

dataset = Active_dataset(x,y, init_size=20)
model = torch.nn.Linear(2,2, bias=True)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1)

# do fit
x_train, y_train = dataset.get_active_data()

for epoch in range(50):
    y_hat = model(x_train)

    loss = criterion(y_hat, y_train)
    loss.backward()

    print(model.weight)
    print(f"Epoch: {epoch} \t Loss: {loss.item():.2f}")
    optimizer.step()
    optimizer.zero_grad()

    draw(x_train, y_train, y_hat)

    # model fits, but data are not linearly separable

#
# # choose data
#
# y_hat = clf.predict(x)
# orig_acc = accuracy(y_hat, y)
#
# print(f"Original Accuracy: {orig_acc * 100:.2f} %")
#
# for i in range(nbr_init_samples + valid_samples, len(x)):
#     tmp_train_data = np.concatenate((train_data, x[i][None, :]))
#     tmp_train_labels = np.append(train_labels, y[i])
#     clf.fit(tmp_train_data, tmp_train_labels)
#     y_hat_add = clf.predict(x)
#     new_accuracy = accuracy(y_hat_add, y)
#     print(f"Iteration: {i} \t After added sample Accuracy: {new_accuracy * 100:.2f} %")
#
#     # jak moc to nici predesle samply
#     y_hat_tmp = clf.predict(tmp_train_data[:valid_samples])
#     past_acc = accuracy(y_hat_tmp, tmp_train_labels[:valid_samples])
#     print(f"Past Acc: {past_acc:.2f}") # choose samples by this
#
# # Retrieve the model parameters.
# b = clf.intercept_[0]
# w1, w2 = clf.coef_.T
# # Calculate the intercept and gradient of the decision boundary.
# c = -b/w2
# m = -w1/w2
#
# # Plot the data and the classification with the decision boundary.
# xmin, xmax = x[:,0].min(), x[:,0].max()
# ymin, ymax = x[:,1].min(), x[:,1].max()
# xd = np.array([xmin, xmax])
# yd = m*xd + c
# plt.plot(xd, yd, 'k', lw=1, ls='--')
# plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
# plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)
#
# # Whole dataset
# plt.scatter(*X[Y==0].T, s=8, alpha=0.5)
# plt.scatter(*X[Y==1].T, s=8, alpha=0.5)
# # train dataset
# plt.scatter(*train_data.T, s=12, alpha=0.7)
#
# plt.xlim(xmin, xmax)
# plt.ylim(ymin, ymax)
# plt.ylabel(r'$x_2$')
# plt.xlabel(r'$x_1$')
#
# plt.show() # draw
#
# acc = np.mean(y == clf.predict(x))
# print(f"Accuracy: {acc * 100:.2f} %")
