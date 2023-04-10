# %% [markdown]
# # Analysis and prediction of total geographical land use
# 
# <!-- <center><img src= "https://www.newfoodmagazine.com/wp-content/uploads/shutterstock_1118643128-scaled.jpg" alt ="crops" style="width:400px;"></center><br> -->

# %% [markdown]
# <!-- - Machine Learning has the capability to effectively analyze soil data, such as moisture level, temperature, and chemical composition, which have a significant impact on the growth of crops and the health of livestock.
# 
# - In the field of agriculture, this technology allows for precise cultivation of crops, where each plant and animal can be treated individually, leading to more effective decisions by farmers.
# 
# - By leveraging Machine Learning, it is possible to develop methods to predict crop yields and assess the quality of crops on a per-species basis, thus making it possible to detect crop diseases and weed infestations which were previously impossible. -->

# %%
# Disable warnings

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


# %%
# Importing the neccessary libraries

import numpy as np
import pandas as pd


# %% [markdown]
# # Ensemble learning on the Kaggle dataset
# 
# Dataset taken from Kaggle, present [here](./datasets/Crop_recommendation.csv)

# %%
# Reading the dataset
df = pd.read_csv("./datasets/crop_recommendation.csv")


# %%
df.head()


# %%
df.describe()


# %% [markdown]
# Colnames is the name of all columns in the datset

# %%
colnames = list(df.columns)
rownums = len(df[colnames[0]])


# %%
print("Columns in the dataset are", ", ".join(colnames))
print("The number of rows is",rownums)


# %% [markdown]
# Let us create separate test and training dataframes from the total dataset. We shuffle the rows first since it is ordered by crops, and we want to include all crops in training.
# 
# Training dataset = First 2000 rows
# Testing dataset = 200 rows after that

# %%
df = df.sample(frac=1).reset_index(drop=True)

size_of_training = rownums - rownums // 10

print(
    f"Length of training dataset is {size_of_training}, length of testing is {rownums - size_of_training}"
)

training_df = df.iloc[:size_of_training, :]
testing_df = df.iloc[size_of_training:, :]


# %%
training_df.head()


# %%
feature_variables = colnames.copy()
feature_variables.remove("label")

target_labels = list(set(training_df["label"]))

print("Feature variables are", ", ".join(feature_variables))
print("Target labels are ", ", ".join(target_labels))


# %% [markdown]
# Feature variables are everything except the name of the crops
# 
# Feature matrix is the values of all the feature variables in a matrix format

# %%
feature_matrix = []

for i in range(len(training_df[colnames[0]])):
    _ = []

    for feature_variable in feature_variables:
        _str = training_df.at[i, feature_variable]

        val = int(_str) if int(_str) == float(_str) else float(_str)

        _.append(val)
    feature_matrix.append(_)


# %%
target_matrix = [
    training_df.at[i, "label"] for i in range(len(training_df[colnames[0]]))
]


# %%
print(
    "Feature & target matrix:",
    f"{feature_matrix[0]} {target_matrix[0]}",
    f"{feature_matrix[1]} {target_matrix[1]}",
    f"{feature_matrix[2]} {target_matrix[2]}",
    f"{feature_matrix[3]} {target_matrix[3]}",
    f"{feature_matrix[4]} {target_matrix[4]}",
    "and so on",
    sep="\n",
)


# %%
# Modules for making a Voter
from sklearn.ensemble import VotingClassifier as Voter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# %%
# Using X,Y naming convention as input output
X = feature_matrix
Y = target_matrix

# Base classifiers
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = LogisticRegression(random_state=1)


# %%
# Voting classifier
ensemble_clf = Voter(
    estimators=[("dt", clf1), ("rf", clf2), ("lr", clf3)], voting="hard"
)

# Fitting your voting classifier to the data (feature_matrix and target_matrix)
ensemble_clf.fit(X, Y)


# %%
print(
    ensemble_clf.predict(
        [[60, 42, 23, 19.32666088, 68.03449300000001, 6.192360002999999, 84.22969177]]
    )
)


# %% [markdown]
# ## Checking accuracy of model using testing dataset (200 rows) after building the ensemble voter on training dataset (2000 rows)

# %%
testing_df.head()


# %%
testing_df_matrix = []


for i in range(len(testing_df[colnames[0]])):
    _ = []

    for feature_variable in feature_variables:
        _str = testing_df.at[size_of_training + i, feature_variable]

        val = int(_str) if int(_str) == float(_str) else float(_str)

        _.append(val)
    testing_df_matrix.append(_)

# %%
prediction_output = ensemble_clf.predict(testing_df_matrix)
count = 0
for i, output in enumerate(prediction_output):
    if output == testing_df.at[size_of_training + i, "label"]:
        count += 1
print(f"Accuracy is {round((count/(rownums - size_of_training))*100,3)}%")


# %% [markdown]
# # Exploratory Data Analysis and Data Visualisation

# %%
# Necessary libraries for visualisation

import seaborn
import matplotlib.pyplot
# %matplotlib inline

# %% [markdown]
# We can use a heatmap to check null/missing values

# %%
seaborn.heatmap(df.isnull(), cmap="coolwarm")
matplotlib.pyplot.show()


# %% [markdown]
# Let's have a closer look at the distribution of temperature and ph.
#     
# It is symmetrical and bell shaped, showing that trials will usually give a result near the average, but will occasionally deviate by large amounts. It's also fascinating how these two really resemble each other!

# %%
matplotlib.pyplot.figure(figsize=(12, 5))

matplotlib.pyplot.subplot(1, 2, 1)
# seaborn.distplot(df_setosa['sepal_length'],kde=True,color='green',bins=20,hist_kws={'alpha':0.3})

seaborn.distplot(df["temperature"], color="purple", bins=15, hist_kws={"alpha": 0.2})

matplotlib.pyplot.subplot(1, 2, 2)
seaborn.distplot(df["ph"], color="green", bins=15, hist_kws={"alpha": 0.2})


# %% [markdown]
# A quick check if the dataset is balanced or not.
# 
# If it is imbalanced, we will need to  downsample targets which are more frequent.

# %%
seaborn.countplot(y="label", data=df, palette="plasma_r")


# %% [markdown]
# There does not seem to be an imbalance.

# %% [markdown]
# ### Master plot to visualize the diagonal distribution between two features for all the combinations!

# %% [markdown]
# It is useful to help see how classes differ from each other in a particular space.

# %%
seaborn.pairplot(df, hue="label")


# %% [markdown]
# When it rains, average rainfall is high  and temperature is mildly chill (less than 30'C).
# 
# Rain affects soil moisture which affects ph of the soil. Here are the crops which are likely to be planted during this season.
# 
# Rice needs <b> heavy rainfall (>200 mm)</b> and a <b>humidity above 80%</b>.
# 
# Coconut is a tropical crop. It needs high humidity therefore explaining massive exports from areas like Kerala and Tamil Nadu, which are few coastal areas around the country.

# %%
seaborn.jointplot(
    x="rainfall",
    y="humidity",
    data=df[(df["temperature"] < 30) & (df["rainfall"] > 120)],
    hue="label",
)


# %% [markdown]
# This graph shows average values of both potassium (K) and nitrogen (N) (>50).
# 
# Fruits which have high nutrients typically has consistent potassium values.

# %%
seaborn.jointplot(x="K", y="N", data=df[(df["N"] > 40) & (df["K"] > 40)], hue="label")


# %% [markdown]
# Pairplot between `humidity` and `K` (potassium levels in the soil.)
# 
# Using `seaborn.jointplot()` for bivariate analysis, we plot `humidity` and `K` levels based on Label type.
# 
# It further generates frequency distribution of classes with respect to features

# %%
seaborn.jointplot(x="K", y="humidity", data=df, hue="label", size=8, s=30, alpha=0.7)


# %%
seaborn.boxplot(y="label", x="ph", data=df)


# %% [markdown]
# As visible, `ph` values are critical when it comes to soil. A stability between 6 and 7 is preffered

# %%
seaborn.boxplot(y="label", x="P", data=df[df["rainfall"] > 150])


# %% [markdown]
# One thing we found during our exploratory analysis stage is that the Phosphorous (`P`) levels are quite differentiable when it rains heavily (above 150 mm).

# %% [markdown]
# #### Further analyzing phosphorous levels.
# 
# When humidity is less than 65, almost same phosphor levels(approx 14 to 25) are required for 6 crops which could be grown just based on the amount of rain expected over the next few weeks.

# %%
seaborn.lineplot(data=df[(df["humidity"] < 65)], x="K", y="rainfall", hue="label")


# %% [markdown]
# # Pre-processing for ML Model

# %%
c = df.label.astype("category")
targets = dict(enumerate(c.cat.categories))
df["target"] = c.cat.codes

y = df.target
X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]


# %% [markdown]
# **Correlation visualization between features. We can see how Phosphorous levels and Potassium levels are highly correlated.**

# %%
seaborn.heatmap(X.corr())


# %% [markdown]
# # FEATURE SCALING
# **Feature scaling is required before creating training data and feeding it to the model.**
# 
# As we saw earlier, two of our features (temperature and ph) are gaussian distributed, therefore scaling them between 0 and 1 with MinMaxScaler.

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# we must apply the scaling to the test set as well that we are computing for the training set
X_test_scaled = scaler.transform(X_test)


# %% [markdown]
# # MODEL SELECTION
# 
# ## KNN Classifier for Crop prediction. 
# <hr>

# %%
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
knn.score(X_test_scaled, y_test)


# %% [markdown]
# ### Confusion Matrix

# %%
from sklearn.metrics import confusion_matrix

mat = confusion_matrix(y_test, knn.predict(X_test_scaled))
df_cm = pd.DataFrame(mat, list(targets.values()), list(targets.values()))
seaborn.set(font_scale=1.0)  # for label size
matplotlib.pyplot.figure(figsize=(12, 8))
seaborn.heatmap(df_cm, annot=True, annot_kws={"size": 12}, cmap="terrain")


# %% [markdown]
# ### Let's try different values of n_neighbors to fine tune and get better results

# %%
k_range = range(1, 11)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    scores.append(knn.score(X_test_scaled, y_test))

matplotlib.pyplot.xlabel("k")
matplotlib.pyplot.ylabel("accuracy")
matplotlib.pyplot.scatter(k_range, scores)
matplotlib.pyplot.vlines(k_range, 0, scores, linestyle="dashed")
matplotlib.pyplot.ylim(0.96, 0.99)
matplotlib.pyplot.xticks([i for i in range(1, 11)])


# %% [markdown]
# ## Classification using Support Vector Classifer (SVC)
# <hr>

# %%
from sklearn.svm import SVC as SupportVectorClassifier

svc_poly = SupportVectorClassifier(kernel="rbf").fit(X_train_scaled, y_train)
print("Rbf Kernel Accuracy: ", svc_poly.score(X_test_scaled, y_test))

svc_linear = SupportVectorClassifier(kernel="linear").fit(X_train_scaled, y_train)
print("Linear Kernel Accuracy: ", svc_linear.score(X_test_scaled, y_test))

svc_poly = SupportVectorClassifier(kernel="poly").fit(X_train_scaled, y_train)
print("Poly Kernel Accuracy: ", svc_poly.score(X_test_scaled, y_test))


# %%
# Increase the accuracy by parameter tuning.


from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

parameters = {
    "C": np.logspace(-3, 2, 6).tolist(),
    "gamma": np.logspace(-3, 2, 6).tolist(),
}
# 'degree': np.arange(0,5,1).tolist(), 'kernel':['linear','rbf','poly']

model = GridSearchCV(
    estimator=SupportVectorClassifier(kernel="linear"),
    param_grid=parameters,
    n_jobs=-1,
    cv=4,
)
model.fit(X_train, y_train)


# %%
print(model.best_score_)
print(model.best_params_)


# %% [markdown]
# - Liner kernel seems to be giving satisfactory results, but using fine tuning increases the computation.
# - The accuracy can be increased in poly-kernel by tweaking parameters, but it leads to intensive overfitting.
# - RBF > linear kernel result wise.
# - Best kernel - <b>Poly kernel</b> (by a small margin).

# %% [markdown]
# ## Classifying using decision tree
# <hr>

# %%
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
clf.score(X_test, y_test)


# %% [markdown]
# ### Let's visualize the import features which are taken into consideration by decision trees.

# %%
matplotlib.pyplot.figure(figsize=(10, 4), dpi=80)
c_features = len(X_train.columns)
matplotlib.pyplot.barh(range(c_features), clf.feature_importances_)
matplotlib.pyplot.xlabel("Feature importance")
matplotlib.pyplot.ylabel("Feature name")
matplotlib.pyplot.yticks(np.arange(c_features), X_train.columns)
matplotlib.pyplot.show()


# %% [markdown]
# ## Classification using Random Forest.
# <hr>

# %%
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=4, n_estimators=100, random_state=42).fit(
    X_train, y_train
)

print("RF Accuracy on training set: {:.2f}".format(clf.score(X_train, y_train)))
print("RF Accuracy on test set: {:.2f}".format(clf.score(X_test, y_test)))


# %% [markdown]
# #### Classification
# 
# Let's use `yellowbrick` for classification report. It is great for visualizing in a tabular format.

# %%
from yellowbrick.classifier import ClassificationReport

classes = list(targets.values())
visualizer = ClassificationReport(clf, classes=classes, support=True, cmap="Blues")

visualizer.fit(X_train, y_train)  # Fit the visualizer and the model
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()


# %% [markdown]
# ## Classification using Gradient Boosting
# <hr>

# %%
from sklearn.ensemble import GradientBoostingClassifier

grad = GradientBoostingClassifier().fit(X_train, y_train)
print("Gradient Boosting accuracy : {}".format(grad.score(X_test, y_test)))



