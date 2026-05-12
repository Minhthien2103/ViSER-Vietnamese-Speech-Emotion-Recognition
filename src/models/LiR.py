from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np

class LinModel:
    """
    Linear Regression "ép" làm phân loại cho ViSER.
    - Vẫn dùng StandardScaler qua Pipeline.
    - Hàm predict tự động làm tròn số thực về nhãn nguyên (0, 1, 2, 3).
    """

    def __init__(self, fit_intercept=True, n_jobs=-1):
        self.fit_intercept = fit_intercept
        
        # Sử dụng Pipeline để gom chung bước scale và model lại với nhau
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('linreg', LinearRegression(
                fit_intercept=fit_intercept,
                n_jobs=n_jobs
            ))
        ])

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        # 1. Dự đoán ra số thực liên tục (ví dụ: 1.2, 2.8, -0.5)
        raw_preds = self.model.predict(X)
        
        # 2. Làm tròn về số nguyên gần nhất
        rounded_preds = np.round(raw_preds)
        
        # 3. Cắt cụt (clip) các giá trị nằm ngoài khoảng [0, 3] 
        # (Đảm bảo không có nhãn -1 hay 4, 5 bị lọt ra)
        final_preds = np.clip(rounded_preds, 0, 3).astype(int)
        
        return final_preds

    def score(self, X, y):
        # Vì Linear Regression mặc định trả về R2 score cho hồi quy,
        # ta phải ghi đè lại để nó tính Accuracy Score cho phân loại.
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def find_best_params(self, X_train, y_train, X_test, y_test):
        """
        Grid search trên fit_intercept (vì LinReg không có C hay max_iter), trả về best params.
        """
        # Linear Regression hầu như không có hyperparameter. 
        # Ta thử nghiệm việc có dùng intercept (sai số tĩnh) hay không.
        intercept_range = [True, False]

        best_acc = 0
        best_intercept = self.fit_intercept

        print(f"  Grid search: fit_intercept={intercept_range}")

        for intercept_val in intercept_range:
            # Cập nhật params cho LinearRegression bên trong Pipeline
            self.model.set_params(linreg__fit_intercept=intercept_val)
            
            self.fit(X_train, y_train)
            acc = self.score(X_test, y_test)
            
            print(f"  fit_intercept={str(intercept_val):>5} → Accuracy: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_intercept = intercept_val

        # Refit với best params
        self.fit_intercept = best_intercept
        self.model.set_params(linreg__fit_intercept=best_intercept)
        self.fit(X_train, y_train)

        print(f"\n Best fit_intercept={best_intercept} (Accuracy: {best_acc:.4f})")
        return best_intercept