import torch
import matplotlib.pyplot as plt

def my_vis(epoch_values, loss_values, val_loss_values, metric_values, val_metric_values, vgg16, best_metric_coeff):
#loss plot
    fig1, ax1 = plt.subplots()
    ax1.set_title('Model Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Number of epochs')
    ax1.plot(epoch_values, loss_values)
    ax1.plot(epoch_values, val_loss_values)
    ax1.legend(['Train', 'Val'])
    plt.savefig('/content/drive/My Drive/phdtaskresults/loss.png')
    plt.show()

#metric plot
    fig2, ax2 = plt.subplots()
    ax2.set_title('Model F1 Score')
    ax2.set_ylabel('F1 Score')
    ax2.set_xlabel('Number of epochs')
    ax2.plot(epoch_values, metric_values)
    ax2.plot(epoch_values, val_metric_values)
    ax2.legend(['Train', 'Val'])
    plt.savefig('/content/drive/My Drive/phdtaskresults/f1_score.png')
    plt.show()

#network weights save

    op_st_dct = vgg16.state_dict()

    torch.save(op_st_dct, '/content/drive/My Drive/phdtaskresults/weights.pth')

    if(best_metric_coeff==1):
        torch.save(op_st_dct, '/content/drive/My Drive/phdtaskresults/best_weights.pth')
        print('Best Output State Updated')
    else:
        print('Best Output State Retained')

    return