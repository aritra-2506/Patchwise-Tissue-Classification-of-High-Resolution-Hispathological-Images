import torch
import gc
import loss_metric


def my_eval(vgg16, optimizer, loader_vl, no_of_batches_1, no_of_epochs, epoch):
    vgg16.eval()
    epoch_val_loss = 0.0
    epoch_val_f1 = 0.0

    batch_index = 1
    samples = 1
    with torch.set_grad_enabled(False):
        for u, (inputs, targets) in enumerate(loader_vl):
            batch_length = len(inputs)
            optimizer.zero_grad()
            inputs = torch.reshape(inputs, (inputs.shape[1], inputs.shape[2], inputs.shape[3], inputs.shape[4]))

            targets = torch.squeeze(targets)
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = vgg16(inputs)
            val_loss = loss_metric.loss(outputs, targets)

            val_f1 = loss_metric.f1_score(outputs, targets)

            epoch_val_loss = epoch_val_loss + val_loss.item()
            epoch_val_f1 = epoch_val_f1 + val_f1.item()

            print('batch', batch_index, 'of', no_of_batches_1, 'epoch', epoch + 1, 'of', no_of_epochs, 'samples', '(',
                  samples, '-',
                  samples + batch_length - 1, ')', '-', 'val_loss', ':',
                  "%.3f" % round((val_loss.item()), 3), '-', 'val_f1', ':', "%.3f" % round((val_f1.item()), 3))
            batch_index = batch_index + 1
            samples = samples + batch_length

        epoch_val_loss = epoch_val_loss / no_of_batches_1
        epoch_val_f1 = epoch_val_f1 / no_of_batches_1

    del inputs
    del targets
    gc.collect()
    torch.cuda.empty_cache()

    return epoch_val_loss, epoch_val_f1
