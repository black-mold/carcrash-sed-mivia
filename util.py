# from sklearn.metrics import f1_score
import numpy as np
import torch





# # 임시로 추가
# def F1_score(mod, TF):

#         TP = ((mod*TF) != 0).sum()
#         TN = (( (torch.logical_not(mod.type(torch.bool)))*TF) != 0).sum()
#         FP = ((mod*(torch.logical_not(TF.type(torch.bool)))) != 0).sum()
#         FN = (((torch.logical_not(mod.type(torch.bool)))*(torch.logical_not(TF.type(torch.bool)))) != 0).sum()
        

#         P = torch.true_divide(TP,TP+FP + 0.00001)
#         R = torch.true_divide(TP,TP+FN + 0.00001)
#         F1 = torch.true_divide(2*P*R , P+R + 0.00001).item()

#         return F1


def F1_score(mod, TF):
    mod = mod.type(torch.bool)
    TF = TF.type(torch.bool)

    TP = (mod & TF).float().sum()
    FP = (mod & ~TF).float().sum()
    FN = (~mod & TF).float().sum()

    precision = TP / (TP + FP + 1e-6)  # Add small value to prevent division by zero
    recall = TP / (TP + FN + 1e-6)  # Add small value to prevent division by zero

    F1 = 2 * (precision * recall) / (precision + recall + 1e-6)  # Add small value to prevent division by zero

    return F1.item()



def calculate_max_f1score(y_hat, label):
    # import pdb; pdb.set_trace()
    # range of thresholds to evaluate
    thresholds = np.arange(0.05, 0.95, 0.01)

    # store F1 scores
    f1_scores = []

    # iterate over thresholds
    for threshold in thresholds:
        # convert probabilities to binary classification
        y_pred_binary = (y_hat > threshold).int()
        
        # calculate F1 score
        f1 = F1_score(label, y_pred_binary)
        f1_scores.append(f1)

    # print the maximum F1 score and the corresponding threshold
    max_f1_idx = np.argmax(f1_scores)
    print(f"Maximum F1 score: {f1_scores[max_f1_idx]:.4f} at threshold: {thresholds[max_f1_idx]:.2f}")

    return f1_scores[max_f1_idx], thresholds[max_f1_idx]

