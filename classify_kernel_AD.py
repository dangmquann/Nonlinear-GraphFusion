import numpy as np
from umap import UMAP
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='umap')

def classify_kernel_AD(outputs,y_train, PET_Z, verbose=False):

  umap = UMAP(n_components=10, random_state=42)

  X_umap = umap.fit_transform(outputs)
  y = y_train[0,0,:]
  Z_train, Z_val, Z_test = PET_Z[0], PET_Z[1],PET_Z[2]
  Z_train = np.concatenate([Z_train, Z_val], axis=0)
  X_train, X_test, y_train, y_test = X_umap[Z_train,:], X_umap[Z_test,:], y[Z_train], y[Z_test]


  # X_train, X_test, y_train, y_test =  train_test_split(X_umap,y, test_size=0.2, random_state=41)
  # Create a Random Forest classifier
  rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

  # Fit the model to the training data
  rf_classifier.fit(X_train, y_train)

  # Get the number of trees in the forest
  num_trees = len(rf_classifier.estimators_)



  # =================
  # Make predictions on the training data
  y_pred = rf_classifier.predict(X_train)

  # Evaluate the model
  accuracy = accuracy_score(y_train, y_pred)
  if verbose:
    print(f"Accuracy train: {accuracy:.2f}")

    # # Display the classification report
    # print("Classification Report train:")
    # print(classification_report(y_train, y_pred))


  # ==========================
  # Make predictions on the testing data
  y_pred = rf_classifier.predict(X_test)

  # Evaluate the model
  accuracy = accuracy_score(y_test, y_pred)
  conf_matrix = confusion_matrix(y_test, y_pred)

  if verbose:
    print(f"Accuracy test: {accuracy:.2f}")

    # # Display the classification report
    # print("Classification Report test:")
    # print(classification_report(y_test, y_pred))
  return confusion_matrix, accuracy

