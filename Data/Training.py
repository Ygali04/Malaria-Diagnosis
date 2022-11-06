import StartingDataset

X_train, X_val, y_train, y_val = train_test_split(X_reshaped, y, test_size = 0.33)
model = SVC()
model.fit(X_train, y_train)

accuracy = model.score(X_val,y_val)
print(accuracy)

dump(model, "model.joblib")
