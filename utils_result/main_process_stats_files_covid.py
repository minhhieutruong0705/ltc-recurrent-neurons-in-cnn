import glob
import os
import csv

from utils_dictionary import group_dictionaries_by_attribute, get_mean_std_of_dictionaries_by_keys
from stats_files_uitls import get_test_result

if __name__ == '__main__':
    summary_file = "covid-models_summary.csv"
    stats_files = glob.glob("../records/covid*/*/*/*_stats.txt")  # get all statistic files

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
        # append in the result list
        result_by_model[model_name].append(std_dict)
        result_by_model[model_name].append(mean_dict)

    # write csv file
    fieldnames = models_result_list[0].keys()
    with open(summary_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name in sorted(result_by_model.keys()):
            writer.writerows(result_by_model[model_name])
