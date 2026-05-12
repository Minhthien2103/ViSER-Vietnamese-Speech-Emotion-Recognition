from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class LogModel:
    """
    Logistic Regression classifier cho ViSER (Mô hình tuyến tính để phân loại).
    - CẦN StandardScaler (linear models rất nhạy với scale của dữ liệu).
    - Dùng Pipeline để tự động scale dữ liệu mỗi khi fit/predict.
    - Xử lý tốt bài toán phân loại đa lớp (multi-class).
    """

    def __init__(self, C=1.0, max_iter=1000, random_state=42):
        self.C = C
        self.max_iter = max_iter
        
        # Sử dụng Pipeline để gom chung bước scale và model lại với nhau
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(
                C=C,
                max_iter=max_iter,
                random_state=random_state,
                multi_class='multinomial', # Chỉ định rõ đây là bài toán đa lớp
                n_jobs=-1
            ))
        ])

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def find_best_params(self, X_train, y_train, X_test, y_test):
        """
        Grid search trên C (hệ số điều chuẩn) và max_iter, trả về best params.
        """
        # C là tham số regularization (nhỏ = phạt nặng, lớn = ít phạt)
        c_range = [0.01, 0.1, 1.0, 10.0, 100.0]
        # max_iter là số vòng lặp tối đa để model hội tụ
        iter_range = [1000]

        best_acc = 0
        best_c, best_iter = self.C, self.max_iter

        print(f"  Grid search: C={c_range}, max_iter={iter_range}")

        for c_val in c_range:
            for iter_val in iter_range:
                # Cập nhật params cho LogisticRegression bên trong Pipeline
                self.model.set_params(logreg__C=c_val, logreg__max_iter=iter_val)
                
                self.fit(X_train, y_train)
                acc = self.score(X_test, y_test)
                
                print(f"  C={c_val:>6}, max_iter={str(iter_val):>4} → Accuracy: {acc:.4f}")

                if acc > best_acc:
                    best_acc = acc
                    best_c = c_val
                    best_iter = iter_val

        # Refit với best params
        self.C = best_c
        self.max_iter = best_iter
        self.model.set_params(logreg__C=best_c, logreg__max_iter=best_iter)
        self.fit(X_train, y_train)

        print(f"\n Best C={best_c}, max_iter={best_iter} (Accuracy: {best_acc:.4f})")
        return best_c, best_iter