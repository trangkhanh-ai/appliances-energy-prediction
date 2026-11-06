"""ridge_scratch.py - Hồi quy Ridge đa đầu ra tự cài đặt bằng NumPy (phần "cài đặt từ đầu" của Phụ lục A).

Công thức: chuẩn hóa Z = (X - mu) / sd (ddof=0, giống StandardScaler), b = trung bình Y,
           W = (Z'Z + alpha*I)^-1 Z'(Y - b)  -- giải hệ tuyến tính, không tính nghịch đảo trực tiếp.
Chạy kiểm tra đối chiếu với scikit-learn:  python ridge_scratch.py
"""
import numpy as np


class RidgeScratch:
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)

    def fit(self, X, Y):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd == 0] = 1.0                      # cột hằng: tránh chia cho 0
        Z = (X - self.mu) / self.sd
        self.b = Y.mean(axis=0)                          # Z đã tâm hóa => hệ số chặn = trung bình nhãn
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.W = np.linalg.solve(A, Z.T @ (Y - self.b))
        return self

    def predict(self, X):
        return ((np.asarray(X, float) - self.mu) / self.sd) @ self.W + self.b


if __name__ == "__main__":
    from pathlib import Path

    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    import appliances_core as c

    df, _ = c.load_raw(Path(__file__).parent / "data" / "energydata_complete.csv")
    X, Y = c.build_features(df), c.build_labels(df)
    cols = c.FEATURE_SETS["A_ar_calendar"]
    tr = c.origin_mask(df.t, *c.SPLITS["train"], X[cols].to_numpy(), Y)
    te = c.origin_mask(df.t, *c.SPLITS["test"], X[cols].to_numpy(), Y)
    for alpha in (1.0, 100.0):
        mine = RidgeScratch(alpha).fit(X[cols].to_numpy()[tr], Y[tr]).predict(X[cols].to_numpy()[te])
        ref = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X[cols].to_numpy()[tr], Y[tr]).predict(X[cols].to_numpy()[te])
        print(f"alpha={alpha:>5}: sai khác lớn nhất so với scikit-learn = {np.abs(mine - ref).max():.3e}")
