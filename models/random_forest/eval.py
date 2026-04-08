from sklearn.metrics import accuracy_score, recall_score, f1_score, cohen_kappa_score
from test import testPipeline

# Obtain Test Results
model_test, y_test = testPipeline()

# Unpackage Results
A_pred, B_pred, C_pred = model_test
y_A_test, y_B_test, y_C_test = y_test

#-----#

# Evaluate Model 1
A_acc = accuracy_score(y_A_test, A_pred)
A_rec = recall_score(y_A_test, A_pred, average='macro')
A_f1 = f1_score(y_A_test, A_pred, average='macro')
A_coh = cohen_kappa_score(y_A_test, A_pred)

print(f"""
      ---Model 3---
        Accuracy : {A_acc}
        Recall   : {A_rec}
        F1-Score : {A_f1}
        Coh Kap  : {A_coh}
      """)

#-----#

# Evaluate Model 2
B_acc = accuracy_score(y_B_test, B_pred)
B_rec = recall_score(y_B_test, B_pred, average='macro')
B_f1 = f1_score(y_B_test, B_pred, average='macro')
B_coh = cohen_kappa_score(y_B_test, B_pred)

print(f"""
      ---Model 3---
        Accuracy : {B_acc}
        Recall   : {B_rec}
        F1-Score : {B_f1}
        Coh Kap  : {B_coh}
      """)

#-----#

# Evaluate Model 3
C_acc = accuracy_score(y_C_test, C_pred)
C_rec = recall_score(y_C_test, C_pred, average='macro')
C_f1 = f1_score(y_C_test, C_pred, average='macro')
C_coh = cohen_kappa_score(y_C_test, C_pred)

print(f"""
      ---Model 3---
        Accuracy : {C_acc}
        Recall   : {C_rec}
        F1-Score : {C_f1}
        Coh Kap  : {C_coh}
      """)

#-----#

print("----- End-----")