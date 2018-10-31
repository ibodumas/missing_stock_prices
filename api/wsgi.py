from sklearn.externals import joblib
safed_model = joblib.load('spline.joblib')
safed_model.predict(88)