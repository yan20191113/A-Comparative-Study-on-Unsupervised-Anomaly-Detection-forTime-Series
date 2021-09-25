# A Comparative Study on Unsupervised Anomaly Detection for Time Series: Experiments and Analysis
## Introduction
This is a repo for our paper "A Comparative Study on Unsupervised Anomaly Detection for Time Series: Experiments and Analysis". We release the source codes of all the models reported in our paper.

## Requirements
Platform and software.
* Linux Ubuntu 16.04
* Python 3.7 (Anaconda is recommended.)

Python Packages and its version: 
* torch == 1.8.1
* scikit-learn == 0.24.1
* numpy == 1.19.5
* scipy == 1.6.2
* pandas == 1.2.4

## Structure
The implementation of models is in the top-level of AD-Framework-main, where the filename is such as baseline_*.py (\* is the model name). For example, baseline_2DCNN.py represents the implementations of model 2DCNN. The directory _utils_ contains some tool functions, which are called by the model. For example, utils/data_provider.py contains the functions for reading and preprocessing datasets, which are called by all the model.

## Framework
We implements a framework, which can involve almost all the existing models for time series anomaly detection. Specifically, this framework can be described as follows.
```python

class Model(nn.Module):
    def __init__(self, ...):
        # define the model structure
        ...
    def forward(self, ...):
        # define the forward propagate
        ...
    def fit(self, ...):
        # define the training strategy
        ...

def RunModel(train_filename=train_file, test_filename=test_file, label_filename=label_file, config=config, ratio=args.ratio):
    ...
    # reading datasets
    train_data, abnormal_data, abnormal_label = read_dataset(train_filename, test_filename, label_filename,
                normalize=True, file_logger=file_logger, negative_sample=negative_sample, ratio=ratio)
    # preprocessing the datasets
    ...
    # instantiate the model
    model = Model(...)
    ...
    # fit the model with given data and arguments
    outputs = model.fit(...)
    ...
    # calculate the anomaly scores for each time step
    error = ...
    ...
    # get anomaly results by three autothreshold methods, i.e., SD, MAD, and IQR.
    SD_y_hat = ...
    MAD_y_hat = ...
    IQR_y_hat = ...
    ...
    # calculate the metrics, such as ROC_AUC, Precision, Recall and so on.
    ...
    # return the performance of this model
    return metrics_result

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    ...
    # set config and logger
    ...
    # run the model with configurations
    metrics_result = RunModel(train_filename=train_file, test_filename=test_file, label_filename=label_file,
                                      config=config, ratio=args.ratio)
    # save the results
    result_dataframe = make_result_dataframe(metrics_result)
    ...
```
All the implementations of the model in this repo, such as 2DCNN, RAE, RN, follow this paradigm.

## Usage
To run the codes, only one thing you should do is setting the correct data path in utils/data_provider.py file.
```python
# around in 20 line
...
datasets_path = Path("../datasets/")
dataset2path = {
    "MSL": {},
    "SMAP": {},
    "SMD": {},
    "NAB": {},
    "AIOps": {},
    "Credit": {},
    "ECG": {},
    "nyc_taxi": {},
    "SWAT": {},
    "Yahoo": {}
}
for dataset in dataset2path:
    dataset2path[dataset]["train"] = datasets_path / "train" / dataset
    dataset2path[dataset]["test"] = datasets_path / "test" / dataset
    dataset2path[dataset]["test_label"] = datasets_path / "test_label" / dataset
...
```
where _dataset2path_ is the dictionary to convert the dataset string to actual path of dataset. For example, suppose the path of MSL are "../datasets/train/MSL/", "../datasets/test/MSL" and "../datasets/test_label/MSL/", which contain training data, testing data, and testing label, respectively. So the _dataset2path_ should be set as follows.
```python
dataset2path["MSL"]["train"] = Path("../datasets/train/MSL/")
dataset2path["MSL"]["test"] = Path("../datasets/test/MSL/")
dataset2path["MSL"]["test_label"] = Path("../datasets/test_label/MSL/")
```
After that, you can run any model directly by choosing the datasets you want to run with. 

For example, assuming you want to run CAE in MSL. Firstly, you should set the running datasets in _baseline_CAE.py_ around in 450 lines.
```python
# Before
...
for registered_dataset in ["Yahoo"]:
    ...

# After
...
for registered_dataset in ["MSL"]:
    ...
```
Then, you can run the model with default hyper-parameters as follows.
```shell script
python baseline_CAE.py
```
Of course, you can run with other hyper-parameters by changing the running command.
```shell script
python baseline_CAE.py --lr 0.002
```
where the learning rate is changed from 0.001 to 0.002. For more details of setting hyper-parameters, you should see the codes about `parser`.