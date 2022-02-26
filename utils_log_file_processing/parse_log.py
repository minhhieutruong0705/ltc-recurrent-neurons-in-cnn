def parse_log(log_file):
    # init data holders
    train_data = {"Loss": [], "Acc": [], "F1": [], "Dice": [], "Pre": [], "Re": [], "TP": [], "TN": [], "FP": [],
                  "FN": []}
    val_data = {"Loss": [], "Acc": [], "F1": [], "Dice": [], "Pre": [], "Re": [], "TP": [], "TN": [], "FP": [],
                "FN": []}
    test_data = {"Loss": [], "Acc": [], "F1": [], "Dice": [], "Pre": [], "Re": [], "TP": [], "TN": [], "FP": [],
                 "FN": []}
    data_dict = {
        "[TRAIN]": train_data,
        "[EVAL]": val_data,
        "[TEST]": test_data
    }

    # parse data
    with open(log_file, 'r') as f:
        lines = f.read()
    lines = lines.split('\n')
    for line in lines:
        # split into metrics
        fields = line.split(',')
        log_mode = fields[0].split(' ')[0]
        if log_mode != "":
            for field in fields:
                field = field.split(':')
                metric = field[-2].strip()
                metric_value = float(field[-1].strip())
                # check keys before append new data to dictionary
                assert log_mode in data_dict.keys()
                assert metric in data_dict[log_mode].keys()
                data_dict[log_mode][metric].append(metric_value)

    # validate data reading
    if len(data_dict["[TRAIN]"]) != len(data_dict["[EVAL]"]):
        print("[ERROR] Train size and validation size mismatch!")

    return data_dict
