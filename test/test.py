from autoctr import Recommender

# rec = Recommender (data_path='/Users/apple/AutoCTR project/dataset/criteo_2m.csv', target='label', sep=',', frac=0.03)
rec = Recommender (data_path='/Users/apple/AutoCTR project/dataset/Movielens/ml-100k/u.data', target='ratings', sep="\\s+")

rec.get_pipeline (pre_train=0, batch_size=1024, max_evals=10,  pre_train_epoch=2, pop_size=40, epochs=10, tuner="bayesian")

# rec = Recommender (data_path='/Users/apple/AutoCTR project/dataset/Book-Crossing/BX-Book-Ratings.csv', target='Book-Rating', sep=";")

# rec.get_pipeline (pre_train=0, batch_size=1024, max_evals=10,  pre_train_epoch=5, tuner='random', epochs=10)
# rec.get_pipeline (pre_train=1, batch_size=1024,  pre_train_epoch=5)

# import torch

# model = torch.load ("/Users/apple/AutoCTR project/AutoCTR/PKL/bayesian/DeepFM.pth")
