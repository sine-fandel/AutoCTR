from autoctr.preprocessor.inputs import Input
import pandas as pd
from autoctr.models import *
from sklearn.metrics import log_loss, roc_auc_score

data_path='/Users/apple/Downloads/tf-experiment/criteo_sample.txt'

profiling = Input (data_path=data_path, sep=",")

train, test, train_model_input, test_model_input, linear_feature_columns, dnn_feature_columns = profiling.preprocessing ()
model = DeepFM (linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, task='binary', l2_reg_embedding=1e-5, device='cpu')
model.compile ("adagrad", "binary_crossentropy", metrics=["binary_crossentropy", "auc"], )
model.fit (train_model_input, train['label'].values, batch_size=32, epochs=10, verbose=2, validation_split=0.8)
pred_ans = model.predict (test_model_input, 256)

print ("test LogLoss", round (log_loss(test['label'].values, pred_ans), 4))
print ("test AUC", round (roc_auc_score(test['label'].values, pred_ans), 4))


	