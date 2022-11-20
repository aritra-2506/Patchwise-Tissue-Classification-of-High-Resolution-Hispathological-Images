import torch.nn as nn
from torchvision import models
import torch
import gc
import torch.nn.functional as nnf
import numpy as np

def my_test(loader_tst, optimizer):
#network reload and weights reload
    vgg16 = models.vgg16(pretrained=True)
    vgg16.classifier[6] = nn.Linear(4096, 3)
    vgg16.cuda()

    vgg16.load_state_dict(torch.load('/content/drive/My Drive/phdtaskresults/best_weights.pth'))

    vgg16.eval()


    with torch.set_grad_enabled(False):
        c_list = []
        for u, (inputs, targets) in enumerate(loader_tst):
            optimizer.zero_grad()
            inputs = torch.reshape(inputs, (inputs.shape[1], inputs.shape[2], inputs.shape[3], inputs.shape[4]))
            inputs = inputs.cuda()
            outputs = vgg16(inputs)

            outputs = outputs.cpu()
            prob = nnf.softmax(outputs, dim=1)
            top_p1, top_class1 = prob.topk(3, dim=1)
            q = top_p1.shape[0]
            r = top_p1.shape[1]
            b_list = []
            for m in range(q):
                a_list = []
                for n in range(r):
                    if(top_p1[m][n]>0.33):
                        a_list.append(top_class1[m][n])
                    else:
                        a_list.append(100)
                a_array = np.asarray(a_list)
                b_list.append(a_array)
            b_array = np.asarray(b_list)
            c_list.append(b_array)
        c_array = np.asarray(c_list)
    pred_array = c_array


    del inputs
    del targets
    gc.collect()
    torch.cuda.empty_cache()
    return pred_array





