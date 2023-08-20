# Pickle saving testing
import pickle
A = [1,2,3,4,5]
B = [6,7,8,9,10]

model_logdir = "/Users/xaviermootoo/Desktop/School Docs/"
stats_logdir = "/Users/xaviermootoo/Desktop/School Docs/"


# Open the file in write-binary mode and use pickle.dump to save the object
with open(model_logdir + "model_test.pickle", 'wb') as f:
    pickle.dump(A, f)
with open(stats_logdir + "stats_test.pickle", 'wb') as f:
    pickle.dump(B, f)