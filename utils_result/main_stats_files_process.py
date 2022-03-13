import glob
import os
import csv

from utils_dictionary import group_dictionaries_by_attribute, get_mean_std_of_dictionaries_by_keys


def get_test_result(file_path):
    test_result_dict = {}  # dictionary to store test result of a model
    metric_names = []  # name of evaluational metrics

    # get test context
    dataset_name, model_name, run_id, _ = os.path.basename(file_path).split('_')
    test_result_dict.update({
        "dataset": dataset_name,
        "model": model_name,
        "run_id": run_id
    })

    # get data
    with open(file_path, 'r') as f:
        test_string = f.readlines()[-1]
    # split data into fields
    test_data = test_string.split(', ')

    # process header to get best validated epoch and loss value
    test_header = test_data.pop(0)
    test_header = test_header.replace("[TEST] ", '').split(': ')
    test_result_dict.update({
        "best_valid": test_header[0],
        test_header[1]: float(test_header[2])
    })
    metric_names.append(test_header[1])  # loss

    # update test result with testing metrics
    for metric in test_data:
        metric_name, metric_score = metric.split(': ')
        test_result_dict.update({metric_name: float(metric_score)})
        metric_names.append(metric_name)
    return test_result_dict, metric_names


if __name__ == '__main__':
    summary_file = "summary_models.csv"
    stats_files = glob.glob("../records/*/*/*/*_stats.txt")  # get all statistic files

    # get all the models result
    models_result_list = []
    metric_names = []
    for stats_file in stats_files:
        model_result, metric_names = get_test_result(file_path=stats_file)
        models_result_list.append(model_result)

    # group by model
    result_by_model = group_dictionaries_by_attribute(in_dict_list=models_result_list, key="model")

    # get mean value of each model
    for model_name in result_by_model.keys():
        mean_dict, std_dict = get_mean_std_of_dictionaries_by_keys(
            in_dict_list=result_by_model[model_name],
            value_keys=metric_names,
            header_keys=["dataset", "model"],
            display_colum="run_id"
        )
        # append in the list
        result_by_model[model_name].append(mean_dict)
        result_by_model[model_name].append(std_dict)

    # write csv file
    fieldnames = models_result_list[0].keys()
    with open(summary_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name in result_by_model.keys():
            writer.writerows(result_by_model[model_name])
