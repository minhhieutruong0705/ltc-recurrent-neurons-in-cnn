import pandas as pd


def group_dictionaries_by_attribute(in_dict_list, key):
    grouped_dict = {}
    for element in in_dict_list:
        key_value = element[key]
        if key_value not in grouped_dict.keys():
            grouped_dict.update({key_value: []})
        grouped_dict[key_value].append(element)
    return grouped_dict


def get_mean_std_of_dictionaries_by_keys(in_dict_list, value_keys, header_keys=[], display_colum=None):
    template = in_dict_list[0]
    # init
    mean_dict = dict.fromkeys(template.keys())
    std_dict = dict.fromkeys(template.keys())
    for header_key in header_keys:
        mean_dict.update({header_key: template[header_key]})
        std_dict.update({header_key: template[header_key]})
    if display_colum is not None:
        mean_dict.update({display_colum: "mean"})
        std_dict.update({display_colum: "std"})

    # get mean and std
    dict_df = pd.DataFrame(in_dict_list)
    values_df = dict_df[value_keys]
    mean_df = values_df.mean().round(decimals=4)
    std_df = values_df.std().round(decimals=4)

    # update in mean and std dict
    mean_dict.update(dict(mean_df))
    std_dict.update(dict(std_df))
    return mean_dict, std_dict
