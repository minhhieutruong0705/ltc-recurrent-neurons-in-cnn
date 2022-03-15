import os


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
