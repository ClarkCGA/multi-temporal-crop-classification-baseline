from pathlib import Path
import tqdm
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

class CropData(Dataset):
    r"""
    Create an iterable dataset of image chips
    Arguments:
        src_dir (str): Path to the folder contains data folders and files.
        usage (str): can be either train, validate, or test.
        dataset_name (str): Name of the training/validation dataset containing 
                            structured folders for image, label, and mask.
        apply_normalization (binary): decides if normalization should be applied.
        trans (list of str): Transformation or data augmentation methods; list 
                             elements could be chosen from:
                             ['v_flip','h_flip','d_flip','rotate','resize']
        split_ratio (float): Number in the range (0,1) that decides on the portion 
                             of samples that should be used for training. 
                             The remaining portion of samples will be assigned to 
                             the 'validation' dataset. Default is 0.8.
        make_deterministic (Binary): If set to True, we seed the numpy randomization 
                                     in splitting the dataset into train and validation 
                                     subfolders.
    Returns:
        A tuple of (image, label) for training and validation but only the image iterable 
        if in the inference phase.
    """

    def __init__(self, src_dir, usage, dataset_name, split_ratio=0.8, 
                 apply_normalization=False, trans=None, **kwargs):

        self.usage = usage
        self.dataset_name = dataset_name
        self.split_ratio = split_ratio
        self.make_deterministic = make_deterministic
        self.apply_normalization = apply_normalization
        self.trans = trans

        assert self.usage in ["train", "validation", "inference"], "Usage is not recognized."

        img_fnames = [Path(dirpath) / f 
                      for (dirpath, dirnames, filenames) in os.walk(Path(src_dir) / self.dataset_name) 
                      for f in filenames 
                      if f.endswith(".tif") and "merged" in f and "_".join(f.split("_")[1:3]) in train_ids]
        img_fnames.sort()

        lbl_fnames = [Path(dirpath) / f 
                      for (dirpath, dirnames, filenames) in os.walk(Path(src_dir) / self.dataset_name) 
                      for f in filenames 
                      if f.endswith(".tif") and "merged" in f and re.search(r"_\d{3}_\d{3}", f)]
        lbl_fnames.sort()

        if self.usage in ["train", "validation"]:

            self.img_chips = []
            self.lbl_chips = []

            total_samples = len(img_fnames)
            indices = np.arange(total_samples)
            split_index = int(total_samples * self.split_ratio)

            np.random.seed(0)
            np.random.shuffle(indices)

            train_indices = indices[:split_index]
            val_indices = indices[split_index:]

            train_img_fnames = [img_fnames[i] for i in train_indices]
            train_lbl_fnames = [lbl_fnames[i] for i in train_indices]

            val_img_fnames = [img_fnames[i] for i in val_indices]
            val_lbl_fnames = [lbl_fnames[i] for i in val_indices]

            if self.usage == "train":
                img_fnames = train_img_fnames
                lbl_fnames = train_lbl_fnames
            else:
                img_fnames = val_img_fnames
                lbl_fnames = val_lbl_fnames

            for img_fname, lbl_fname in tqdm.tqdm(zip(img_fnames, lbl_fnames), 
                                                  total=len(img_fnames)):
                    
                img_chip = load_data(Path(src_dir) / self.dataset_name / img_fname,
                                     apply_normalization=self.apply_normalization, 
                                     is_label=False)
                img_chip = img_chip.transpose((1, 2, 0))

                lbl_chip = load_data(Path(src_dir) / self.dataset_name / lbl_fname, 
                                     is_label=True)

                self.img_chips.append(img_chip)
                self.lbl_chips.append(lbl_chip)
        else:
            pass
        
        print(f"------ {self.usage} dataset with {len(self.img_chips)} patches created ------")


    def __getitem__(self, index):
        """
        Support indexing such that dataset[index] can be used to get 
        the (index)th sample.
        """
        if self.usage in ["train", "validation"]:
            img_chip = self.img_chips[index]
            lbl_chip = self.lbl_chips[index]

            if self.trans and self.usage == "train":
                trans_flip_ls = [m for m in self.trans if "flip" in m]
                if random.randint(0, 1) and len(trans_flip_ls) > 1:
                    trans_flip = random.sample(trans_flip_ls, 1)
                    img_chip, lbl_chip = flip(img_chip, lbl_chip, trans_flip[0])
                    
                if random.randint(0, 1) and "resize" in self.trans:
                    scale_factor = kwargs.get("scale_factor", (0.75, 1.5))
                    img_chip, lbl_chip = re_scale(img_chip, lbl_chip.astype(np.uint8),
                                                  scale=scale_factor, crop_strategy="center")
                    
                if random.randint(0, 1) and "rotate" in self.trans:
                    deRotate = kwargs.get("rotation_degree", (-90, 90))
                    img_chip, lbl_chip = center_rotate(img_chip, lbl_chip, deRotate)
                    
                if random.randint(0, 1) and 'shift_brightness' in self.trans:
                    bshift_subs = kwargs.get("bshift_subs", (3, 3))
                    bshift_gamma_range = kwargs.get("bshift_gamma_range", (0.2, 2.0))
                    patch_shift = kwargs.get("patch_shift", True)
                    img_chip = shift_brightness(img_chip, gamma_range=bshift_gamma_range,
                                                shift_subset=bshift_subs, patch_shift=patch_shift)
            
            label = torch.from_numpy(np.ascontiguousarray(lbl_chip)).long()
            # shape from (H,W,C) --> (C,H,W)
            img_chip = torch.from_numpy(img_chip.transpose((2, 0, 1))).float()

            return img_chip, label
        
        else:
            img_chip = self.img_chips[index]
            img_chip = torch.from_numpy(img_chip.transpose((2, 0, 1))).float()

            return img_chip

    def __len__(self):
        return len(self.img_chips)