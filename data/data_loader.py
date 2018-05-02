
def CreateDataLoader(opt):

    if(opt.medical_data):
        from data.med_data_loader.med_dataset_data_loader import MedDatasetDataLoader
        data_loader = MedDatasetDataLoader()
    else:
        from data.custom_dataset_data_loader import CustomDatasetDataLoader
        data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
