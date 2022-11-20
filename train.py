import torch
import gc
import loss_metric

def my_train(vgg16, optimizer, loader_tr, no_of_batches, no_of_epochs, epoch):
    vgg16.train()
    epoch_loss = 0.0
    epoch_f1 = 0.0

    batch_index = 1
    samples = 1
    for u, (inputs, targets) in enumerate(loader_tr):
        batch_length = len(inputs)
        optimizer.zero_grad()
        inputs = torch.reshape(inputs, (inputs.shape[1], inputs.shape[2], inputs.shape[3], inputs.shape[4]))

        targets = torch.squeeze(targets)

        inputs = inputs.cuda()
        targets = targets.cuda()

        outputs = vgg16(inputs)

        loss = loss_metric.loss(outputs, targets)
        f1 = loss_metric.f1_score(outputs, targets)

        epoch_loss = epoch_loss + loss.item()
        epoch_f1 = epoch_f1 + f1.item()

        print('batch', batch_index, 'of', no_of_batches, 'epoch', epoch + 1, 'of', no_of_epochs, 'samples', '(', samples, '-',
              samples + batch_length - 1, ')', '-', 'loss', ':',
              "%.3f" % round((loss.item()), 3), '-', 'f1', ':', "%.3f" % round((f1.item()), 3))
        batch_index = batch_index + 1
        samples = samples + batch_length

        loss.backward()
        optimizer.step()
    epoch_loss = epoch_loss / no_of_batches
    epoch_f1 = epoch_f1 / no_of_batches

    del inputs
    del targets
    gc.collect()
    torch.cuda.empty_cache()

    return epoch_loss, epoch_f1
