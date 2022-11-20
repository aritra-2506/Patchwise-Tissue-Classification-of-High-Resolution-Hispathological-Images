import numpy as np
import torch.optim as optim
from network import my_net
from data_loader import loaders
from train import my_train
from eval import my_eval
from visualize import my_vis
from test import my_test

batch_size = 1

#data loading
loader_tr = loaders(batch_size, 0)
loader_vl = loaders(batch_size, 1)
loader_tst = loaders(batch_size, 2)
no_of_batches = len(loader_tr)
no_of_batches_1 = len(loader_vl)
no_of_batches_2 = len(loader_tst)

#network
vgg16 = my_net()
vgg16.cuda()

#optimizer
optimizer = optim.Adam(vgg16.parameters(), lr=0.0003, weight_decay=1e-4)
no_of_epochs = 100
best_metric = 0

metric_values, val_metric_values, epoch_values, loss_values, val_loss_values  = ([] for i in range(5))

for epoch in range(no_of_epochs):

#training
    epoch_loss, epoch_f1 = my_train(vgg16, optimizer, loader_tr, no_of_batches, no_of_epochs, epoch)
#validation
    epoch_val_loss, epoch_val_f1 = my_eval(vgg16, optimizer, loader_vl, no_of_batches_1, no_of_epochs, epoch)
#resuls
    print('epoch', epoch + 1, 'of', no_of_epochs, '-', 'epoch_loss', ':',
          "%.3f" % round((epoch_loss), 3), '-', 'epoch_f1', ':', "%.3f" % round((epoch_f1), 3), '-',
          'epoch_val_loss', ':', "%.3f" % round((epoch_val_loss), 3), '-',
          'epoch_val_f1', ':',
          "%.3f" % round((epoch_val_f1), 3))

    metric_values.append(round(epoch_f1, 3))
    val_metric_values.append(round(epoch_val_f1, 3))

    loss_values.append(round(epoch_loss, 3))
    val_loss_values.append(round(epoch_val_loss, 3))

#visualization

    current_metric = round(epoch_val_f1, 3)

    if(current_metric>best_metric):
        best_metric_coeff = 1
        best_metric = current_metric
    else:
        best_metric_coeff = 0

    epoch_values.append(epoch + 1)

    my_vis(epoch_values, loss_values, val_loss_values, metric_values, val_metric_values, vgg16, best_metric_coeff)

    best_metric_value = np.amax(np.asarray(val_metric_values))
    best_loss_value = np.amin(np.asarray(val_loss_values))

    print('Maximum Validation F1 Score', ':', "%.3f" % best_metric_value)
    print('Minimum Validation Loss', ':', "%.3f" % best_loss_value)


print('Finished Training')

#test
pred_array = my_test(loader_tst, optimizer)
print(pred_array)