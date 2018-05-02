import torch.utils.data
from ..base_data_loader import BaseDataLoader

def CreateDataset(opt):
    dataset = None
    from .med_aligned_dataset import MedAlignedDataset
    dataset = MedAlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class MedDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'MedDatasetDataLoader'

    def initialize(self, opt):

        if 'scale_width' not in opt.resize_or_crop:
            raise NotImplementedError("Resizing of medical image data is not supported!")

        if not opt.no_instance:
            raise NotImplementedError("WARNING: Instance maps are untested and not supported with medical image data loading!")

        if opt.load_features:
            raise NotImplementedError("WARNING: Feature map loading is untested and not supported with medical image data loading!")

        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
