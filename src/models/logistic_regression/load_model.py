import numpy as np, joblib, os

def load():
    w = np.load(os.path.join(os.path.dirname(__file__), "weights.npy"))
    b = np.load(os.path.join(os.path.dirname(__file__), "bias.npy"))[0]
    scaler = joblib.load(os.path.join(os.path.dirname(__file__), "../../preprocessing/scaler.pkl"))
    return w, b, scaler

if __name__ == "__main__":
    w,b,scaler = load()
    print("loaded shapes:", getattr(w,'shape',None), b)
