import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
from matplotlib import image
import albumentations as A


# dataset paths

train = "/content/drive/MyDrive/data_007/trainingdata"
val = "/content/drive/MyDrive/data_007/validationdata"
test = "/content/drive/MyDrive/data_007/testdata"

#patchwise splitting
def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))

#data pre-processing
class ImageData(Dataset):
    def __init__(self, data, phase_coeff, phase):
        self.root = data
        self.files = os.listdir(self.root)
        self.files.sort()
        self.aug = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.3, rotate_limit=45, interpolation=1, border_mode=4, always_apply=False, p=0.3),
            A.RandomCrop(25, 25, always_apply=False, p=1.0),
            A.HorizontalFlip(always_apply=False, p=0.5),
            A.VerticalFlip(always_apply=False, p=0.5),
            A.RandomBrightness(limit=0.3, always_apply=False, p=0.5),
            A.RandomContrast(limit=0.3, always_apply=False, p=0.5),
            A.Resize(32, 32),
        ])
        self.phase_coeff = phase_coeff
        self.phase = phase

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        file_name = os.path.join(self.root, self.files[index])

        data = image.imread(file_name)

        inputs = data[0:1024, 0:512, :]
        inputs = np.transpose(inputs, (2, 0, 1))
        inputs_r = inputs[0]
        inputs_g = inputs[1]
        inputs_b = inputs[2]

        inputs_r = blockshaped(inputs_r, 32, 32)
        inputs_g = blockshaped(inputs_g, 32, 32)
        inputs_b = blockshaped(inputs_b, 32, 32)

        inputs_list = []
        inputs_list.append(inputs_r)
        inputs_list.append(inputs_g)
        inputs_list.append(inputs_b)

        inputs_preproc = np.asarray(inputs_list)
        inputs_preproc = np.transpose(inputs_preproc, (1, 0, 2, 3))

        img_array = inputs_preproc


#data augmentation only for training
        if (self.phase_coeff == 1):
            inputs_preproc = np.transpose(inputs_preproc, (0, 2, 3, 1))
            o = inputs_preproc.shape[0]
            img_list = []
            for b in range(o):
                z = inputs_preproc[b]
                transformed = self.aug(image=z, mask=z)
                inputs_ind = transformed['image']
                img_list.append(inputs_ind)
            img_array = np.asarray(img_list)
            img_array = np.transpose(img_array, (0, 3, 1, 2))


        targets_array = data[:, 512:1025, :]
        targets_array = np.transpose(targets_array, (2, 0, 1))
        targets_array = targets_array[2]
        targets_array = targets_array * 255
        targets_array = np.trunc(targets_array)
#test phase
        if(self.phase == 2):
            inputs = torch.from_numpy(img_array)
            targets = inputs
#other phases
        else:
            targets_array = blockshaped(targets_array, 32, 32)
            inputs_newlist = []
            targets_list = []
            targets_index = targets_array.shape[0]
            for i in range(targets_index):
                patch_selection = np.max(targets_array[i])
                if (patch_selection > 0):
                    targets_list.append(patch_selection - 1)
                    targets = np.asarray(targets_list)
                    inputs_newlist.append(img_array[i])
                    inputs = np.asarray(inputs_newlist)

            inputs = torch.from_numpy(inputs)
            targets = torch.from_numpy(targets)
            targets = targets.to(torch.int64)

        return inputs, targets



def loaders(batch_size, phase):
    if (phase == 0):
        dataset = ImageData(train, 1, phase)
    elif (phase == 1):
        dataset = ImageData(val, 0, phase)
    elif (phase == 2):
        dataset = ImageData(test, 0, phase)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers = 4
    )

    return loader