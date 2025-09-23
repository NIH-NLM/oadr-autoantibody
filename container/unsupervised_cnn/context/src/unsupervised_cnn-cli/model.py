# model.py
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import BernoulliRBM

def run_rbm_model(X, output_prefix):
    model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('rbm', BernoulliRBM(n_components=10, learning_rate=0.01, n_iter=20, verbose=True))
    ])
    transformed = model.fit_transform(X)
    np.savetxt(f"{output_prefix}_cnn_latent.csv", transformed, delimiter=",")
    print(f"Saved: {output_prefix}_cnn_latent.csv")
