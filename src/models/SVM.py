from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class SVMModel:
    """
    SVM classifier cho ViSER.
    - Dùng RBF kernel (phù hợp cho dữ liệu phi tuyến như cảm xúc).
    - Tích hợp StandardScaler (SVM nhạy cảm với scale).
    - probability = True để hỗ trợ predict_proba nếu cần.
    """

    def __init__(self, C = 1.0, kernel = 'rbf', gamma = 'scale'):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.scaler = StandardScaler()
        self.model = SVC(
            C = C,
            kernel = kernel,
            gamma = gamma,
            probability = True,
            random_state = 42,
        )


    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        return self


    def predict(self, X):
        X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled)


    def score(self, X, y):
        X_scaled = self.scaler.transform(X)

        return self.model.score(X_scaled, y)


    def find_best_params(self, X_train, y_train, X_test, y_test):
        """
            Grid search trên C và gamma, trả về best params.
        """

        C_range = [0.1, 1, 10, 100]
        gamma_range = ['scale', 'auto', 0.01, 0.001]

        best_acc = 0
        best_C, best_gamma = self.C, self.gamma

        print(f"  Grid search: C={C_range}, gamma={gamma_range}")

        for C in C_range:
            for gamma in gamma_range:
                self.model.set_params(C = C, gamma = gamma)
                self.fit(X_train, y_train)
                acc = self.score(X_test, y_test)
                print(f"  C={C:>5}, gamma={str(gamma):>6} → Accuracy: {acc:.4f}")

                if acc > best_acc:
                    best_acc = acc
                    best_C = C
                    best_gamma = gamma

        # Refit với best params
        self.C = best_C
        self.gamma = best_gamma
        self.model.set_params(C = best_C, gamma = best_gamma)
        self.fit(X_train, y_train)

        print(f"\n Best C={best_C}, gamma={best_gamma} (Accuracy: {best_acc:.4f})")
        return best_C, best_gamma
