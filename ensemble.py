import os
import sys
import json
import pickle
import numpy as np
from sklearn.metrics import classification_report, f1_score


def ensemble_results(specifics: [str]):
    specific_list = [pickle.load(open(specific, "rb")) for specific in specifics]
    # imgname, y, logits, probs
    results = [
        [item[0], item[1], np.array(item[2]), np.array(item[3])]
        for item in specific_list[0]
    ]
    # ensemble
    for specific in specific_list[1:]:
        for i, item in enumerate(specific):
            assert results[i][0] == item[0] and results[i][1] == item[1]
            results[i][2] += item[2]
            results[i][3] += item[3]

    y, logits_p, probs_p = [], [], []
    n = len(specific_list)
    for i in range(len(results)):
        results[i][2] = results[i][2] / n
        results[i][3] = results[i][3] / n
        results[i].append(np.argmax(results[i][2]))
        results[i].append(np.argmax(results[i][3]))
        y.append(results[i][1])
        logits_p.append(results[i][-2])
        probs_p.append(results[i][-1])

    print("logits ensemble")
    print(classification_report(y, logits_p))
    print(f'macro_f1 = {f1_score(y, logits_p, average="macro")}')
    report = classification_report(y, logits_p, output_dict=True)
    json.dump(report, open(f"ensemble_logits.json", "w"), indent=4)
    # target_names = [id2class[i] for i in range(num_classes)]
    # report = {k+(target_names[int(k)] if '0' <= k[0] <= '9' else ''): v for k, v in report.items()}

    print("probability ensemble")
    print(classification_report(y, probs_p))
    print(f'macro_f1 = {f1_score(y, probs_p, average="macro")}')
    report = classification_report(y, probs_p, output_dict=True)
    json.dump(report, open(f"ensemble_probs.json", "w"), indent=4)

    pickle.dump(results, open("ensemble_specific", "wb"))


if __name__ == "__main__":
    ensemble_results(sys.argv[2:])
