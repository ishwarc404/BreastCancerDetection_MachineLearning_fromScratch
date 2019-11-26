from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=21)
neigh.fit(X_train, y_train) 
y_pred_val = neigh.predict(X_val)
print accuracy_score(y_val, y_pred_val)