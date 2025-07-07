from collections import defaultdict
def compute_temporal_iou(pred, gt):
    """ deprecated due to performance concerns
    compute intersection-over-union along temporal axis
    Args:
        pred: [st (float), ed (float)]
        gt: [st (float), ed (float)]
    Returns:
        iou (float):

    Ref: https://github.com/LisaAnne/LocalizingMoments/blob/master/utils/eval.py
    """
    intersection = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])  # not the correct union though
    if union == 0:
        return 0
    else:
        return 1.0 * intersection / union
def temporal_nms(predictions, nms_thd, max_after_nms=100):
    """
    Args:
        predictions: list(sublist), each sublist is [st (float), ed(float), score (float)],
            note larger scores are better and are preserved. For metrics that are better when smaller,
            please convert to its negative, e.g., convert distance to negative distance.
        nms_thd: float in [0, 1]
        max_after_nms:
    Returns:
        predictions_after_nms: list(sublist), each sublist is [st (float), ed(float), score (float)]
    References:
        https://github.com/wzmsltw/BSN-boundary-sensitive-network/blob/7b101fc5978802aa3c95ba5779eb54151c6173c6/Post_processing.py#L42
    """
    if len(predictions) == 1:  # only has one prediction, no need for nms
        return predictions

    predictions = sorted(predictions, key=lambda x: x[2], reverse=True)  # descending order

    tstart = [e[0] for e in predictions]
    tend = [e[1] for e in predictions]
    tscore = [e[2] for e in predictions]
    rstart = []
    rend = []
    rscore = []
    while len(tstart) > 1 and len(rscore) < max_after_nms:  # max 100 after nms
        idx = 1
        while idx < len(tstart):  # compare with every prediction in the list.
            if compute_temporal_iou([tstart[0], tend[0]], [tstart[idx], tend[idx]]) > nms_thd:
                # rm highly overlapped lower score entries.
                tstart.pop(idx)
                tend.pop(idx)
                tscore.pop(idx)
                # print("--------------------------------")
                # print(compute_temporal_iou([tstart[0], tend[0]], [tstart[idx], tend[idx]]))
                # print([tstart[0], tend[0]], [tstart[idx], tend[idx]])
                # print(tstart.pop(idx), tend.pop(idx), tscore.pop(idx))
            else:
                # move to next
                idx += 1
        rstart.append(tstart.pop(0))
        rend.append(tend.pop(0))
        rscore.append(tscore.pop(0))

    if len(rscore) < max_after_nms and len(tstart) >= 1:  # add the last, possibly empty.
        rstart.append(tstart.pop(0))
        rend.append(tend.pop(0))
        rscore.append(tscore.pop(0))

    predictions_after_nms = [[st, ed, s] for s, st, ed in zip(rscore, rstart, rend)]
    return predictions_after_nms
def post_processing_mr_nms(return_list, idx):
    predicted_moments = [[item[0], item[1], item[idx]] for item in return_list]

    predicted_moments = sorted(predicted_moments, key=lambda x: x[2], reverse=True)  # descending order

    nms_thd_list = [0.5, 0.7, 0.9]
    final_predicted_moments = []
    for nms_thd in nms_thd_list:
    
        after_nms_predicted_moments = temporal_nms(
            predicted_moments,
            nms_thd=nms_thd,
            max_after_nms=5
        )
        # 去重
        after_nms_predicted_moments=[item for item in after_nms_predicted_moments if item not in final_predicted_moments]
        
        final_predicted_moments.extend(after_nms_predicted_moments)
        if len(final_predicted_moments) >= 5:
            final_predicted_moments = final_predicted_moments[:5]
            not_enough = False
            break
    if len(final_predicted_moments) < 5:
        miss_number = 5 - len(final_predicted_moments)
        last_prediction = final_predicted_moments[-1]
        final_predicted_moments.extend([last_prediction] * miss_number)
        not_enough = True 

    assert len(final_predicted_moments) == 5

    after_nms_output = [[_item[0], _item[1],_item[2]]
                        for _item in final_predicted_moments]
    return after_nms_output,not_enough
def top1_generator(input_list):
    '''
    input_list: [[start, end, score], [start, end, score], ...]
    return: [[start, end, score, 0, sum_score], ...]
    '''
    ###
    # 1. Compute the center of the proposal
    # 2. Conduct clustering via moment candidate center
    # 3. Create new proposal and score based on the clustered group
    ###

    # 1. Compute the center of the proposal
    center_dict = {}
    for item in input_list:
        center = (item[1] + item[0]) / 2
        center_dict[center] = [item[0], item[1], item[-1]]

    center_list = sorted(list(center_dict.keys()))
    center_cluster_dict = defaultdict(list)

    # 2. Conduct clustering via moment candidate center
    final_idx = len(center_list)  # 3 * max_input
    cur_idx = 0
    distance = 2
    cluster_idx = 0
    center_cluster_dict[cluster_idx].append(center_list[cur_idx])
    cur_idx += 1

    while cur_idx < final_idx:
        current_number = center_list[cur_idx]
        before_number = center_list[cur_idx - 1]
        while current_number - before_number < distance:
            center_cluster_dict[cluster_idx].append(current_number)
            before_number = current_number
            cur_idx += 1
            if cur_idx == final_idx:
                break
            current_number = center_list[cur_idx]

        if cur_idx == final_idx:
            break
        cluster_idx += 1
        center_cluster_dict[cluster_idx].append(current_number)
        cur_idx += 1

    # 3. Create new proposal and score based on the clustered group
    predicted_times_list = []
    for k, v in center_cluster_dict.items():
        temp_score_values_list = [center_dict[item][-1] for item in v]
        total_score = sum(temp_score_values_list)
        import operator
        max_index, max_value = max(enumerate(temp_score_values_list), key=operator.itemgetter(1))
        maximum_score_proposal = center_dict[v[max_index]]

        if len(v) % 2 == 0:
            temp_value_idx = int(len(v) / 2)
            temp_center_value = v[temp_value_idx]
            score1 = center_dict[temp_center_value][-1]
            score2 = center_dict[v[temp_value_idx - 1]][-1]
            if score1 > score2:
                middle_proposal = center_dict[v[temp_value_idx]]
            else:
                middle_proposal = center_dict[v[temp_value_idx - 1]]

        else:
            temp_value_idx = int((len(v) - 1) / 2)
            temp_center_value = v[temp_value_idx]
            middle_proposal = center_dict[temp_center_value]

        new_proposal = [(item1 + item2) / 2 for item1, item2 in zip(middle_proposal, maximum_score_proposal)]
        new_proposal.append(0)
        new_proposal.append(0)
        new_proposal[-1] = total_score
        predicted_times_list.append(new_proposal)

    return sorted(predicted_times_list, key=lambda x: x[-1], reverse=True)