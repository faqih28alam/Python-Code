#reference
#https://www.codegrepper.com/code-examples/python/save+a+machine+learning+h5+model+in+python

# fit the model
model.fit(X_train, y_train)

# save the model
import pickle
pickle.dump(model, open("model.pkl", "wb"))

# load the model
model = pickle.load(open("model.pkl", "rb"))

# use model to predict
y_pred = model.predict(X_input)
