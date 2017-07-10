import pandas as pd
import joblib


from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

filepath = "./model_dt.csv"

print('Loading data')
df = pd.read_csv(filepath, header=0, encoding='latin-1')

# print("Adding features")
# # add binary class or not
# df['BinaryClass'] = df['Classes'].map(lambda x: x == 2)
#
# # add nb class in between 2 to 10
# df['NbClassGt2Lt10'] = df['Classes'].map(lambda x: x > 2 & x <= 10)
#
# # add nb class in between 10 to inf
# df['NbClassGt10'] = df['Classes'].map(lambda x: x > 10)
#
# # add nb_train < 100
# df['NbTrainLt100'] = df['NbTrain'].map(lambda x: x < 100)
#
# # add nb_train > 100 and < 500
# df['NbTrainGt100Lt500'] = df['NbTrain'].map(lambda x: x >= 100 & x < 500)
#
# # add nb_train > 500 and < inf
# df['NbTrainGt500'] = df['NbTrain'].map(lambda x: x >= 500)
#
# # add nb_test < 100
# df['NbTestLt100'] = df['NbTest'].map(lambda x: x < 100)
#
# # add nb_test > 100 and < 500
# df['NbTestGt100Lt500'] = df['NbTest'].map(lambda x: x >= 100 & x < 500)
#
# # add nb_test > 500 and < inf
# df['NbTestGt500'] = df['NbTest'].map(lambda x: x >= 500)
#
# # add sequence length < 100
# df['SequenceLengthLt100'] = df['SequenceLength'].map(lambda x: x <= 100)
#
# # aadd sequence length >= 100 to 512
# df['SequenceLengthGt100Lt512'] = df['SequenceLength'].map(lambda x: x > 100 & x <= 512)
#
# # add nb class in between 10 to inf
# df['SequenceLengthGt512'] = df['SequenceLength'].map(lambda x: x > 512)

# encode type of input data
df['Type'] = df[['Type']].apply(LabelEncoder().fit_transform)

# extract label

label = df['ModelWithAttention']

# drop original data for bucketized rules
#df.drop(['Classes', 'NbTrain', 'NbTest', 'SequenceLength'], axis=1, inplace=True)

# remove class label from train data
df.drop(['ModelWithAttention'], axis=1, inplace=True)

X = df.values
y = label.values

decision_tree = DecisionTreeClassifier(criterion='gini', max_depth=3,
                                       random_state=1024, presort=True)

print("Training model")
decision_tree.fit(X, y)

print("Saving model")
joblib.dump(decision_tree, 'decision_tree.pkl')

print("Computing scores", "\n")
y_preds = decision_tree.predict(X)

accuracy = accuracy_score(y, y_preds)
f1 = f1_score(y, y_preds)

print("Accuracy : ", accuracy)
print("F1 score : ", f1)

print("Confusion Matrix : \n")
print(confusion_matrix(y, y_preds))

try:
    import pydotplus

    feature_names = df.columns.values
    class_names = ['Non Attention Model', 'Attention Model']

    dot_data = export_graphviz(decision_tree, out_file=None, label='root', impurity=False,
                               feature_names=feature_names, class_names=class_names,
                               node_ids=False)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("model_decision.pdf")

except ImportError:
    print("To print the decision tree, pydotplus module dependency must be installed.")
