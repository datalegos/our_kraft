# import pickle
# import pandas as pd
# with open("models/random_forest_model.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("models/label_encoder.pkl", "rb") as f:
#     le = pickle.load(f)

# # print("Model classes:", model.classes_)
# print("Label encoder classes:", le.classes_)

# from sklearn.preprocessing import LabelEncoder

# le_protocol = LabelEncoder()
# X['protocol_type'] = le_protocol.fit_transform(X['protocol_type'])

# le_service = LabelEncoder()
# X['service'] = le_service.fit_transform(X['service'])

# le_flag = LabelEncoder()
# X['flag'] = le_flag.fit_transform(X['flag'])

# import pickle

# with open('le_protocol.pkl', 'wb') as f:
#     pickle.dump(le_protocol, f)
# with open('le_service.pkl', 'wb') as f:
#     pickle.dump(le_service, f)
# with open('le_flag.pkl', 'wb') as f:
#     pickle.dump(le_flag, f)

# X = pd.get_dummies(X, columns=['protocol_type', 'service', 'flag'])
# print(X)
