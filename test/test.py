# from autoctr.preprocessor.inputs import Input
# import pandas as pd
# from autoctr.models import *
# from sklearn.metrics import log_loss, roc_auc_score

# data_path = '/Users/apple/Downloads/tf-experiment/criteo_sample.txt'
# target = 'label'
# profiling = Input (data_path=data_path, sep=",", target=target)

# train, test, train_model_input, test_model_input, linear_feature_columns, dnn_feature_columns = profiling.preprocessing (impute_method='iterative')
# model_list = [DeepFM, AutoInt]

# for Model in model_list :
# 	model = Model (linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, task='binary', l2_reg_embedding=1e-5, device='cpu')
# 	model.compile ("adagrad", "binary_crossentropy", metrics=["binary_crossentropy", "auc"], )
# 	model.fit (train_model_input, train[target].values, batch_size=32, epochs=100, verbose=2, validation_split=0.8)
# 	pred_ans = model.predict (test_model_input, 256)

# 	print ("test LogLoss", round (log_loss(test['label'].values, pred_ans), 4))
# 	print ("test AUC", round (roc_auc_score(test['label'].values, pred_ans), 4))

# from autoctr.process import AutoCTR

# # rec = AutoCTR (data_path='/Users/apple/AutoCTR project/AutoCTR/autoctr/criteo_sample.txt', target='label', sep=",")
# rec = AutoCTR (data_path='/Users/apple/AutoCTR project/dataset/Movielens/ml-100k/u.data', target='ratings', sep="\\s+")
# rec.quality_checking ()
# rec.data_cleaning ()
# rec.preprocessing ()

# rec.run (epochs=100)
# # rec.search ()

from autoctr import Recommender

# rec = Recommender (data_path='/Users/apple/AutoCTR project/dataset/criteo_2m.csv', target='label', sep=',', frac=0.05)
# rec = Recommender (data_path='/Users/apple/AutoCTR project/dataset/Book-Crossing/BX-Book-Ratings.csv', target='Book-Rating', sep=";")
rec = Recommender (data_path='/Users/apple/AutoCTR project/dataset/Movielens/ml-100k/u.data', target='ratings', sep="\\s+", frac=0.1)

# rec.get_pipeline (pre_train=0, batch_size=1024, max_evals=10,  pre_train_epoch=5, tuner='random', epochs=10)
rec.get_pipeline (pre_train=0, batch_size=1024, max_evals=10,  pre_train_epoch=5, epochs=10)
# rec.get_pipeline (pre_train=1, batch_size=1024,  pre_train_epoch=5)
