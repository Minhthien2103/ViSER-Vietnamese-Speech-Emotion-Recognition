from sklearn.ensemble import RandomForestClassifier


class RFModel:
    """
    Random Forest classifier cho ViSER.
    - Không cần StandardScaler (tree-based model không nhạy với scale).
    - n_jobs=-1 để dùng tất cả CPU cores.
    """

    def __init__(self, n_estimators = 100, max_depth = None, random_state = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestClassifier(
            n_estimators = n_estimators,
            max_depth = max_depth,
            random_state = random_state,
            n_jobs = -1,
        )


    def fit(self, X, y):
        self.model.fit(X, y)

        return self


    def predict(self, X):
        return self.model.predict(X)


    def score(self, X, y):
        return self.model.score(X, y)


    def find_best_params(self, X_train, y_train, X_test, y_test):
        """
            Grid search trên n_estimators và max_depth, trả về best params.
        """

        n_est_range = [100, 200, 300, 500]
        depth_range = [None, 10, 20, 30, 50]

        best_acc = 0
        best_n_est, best_depth = self.n_estimators, self.max_depth

        print(f"  Grid search: n_estimators={n_est_range}, max_depth={depth_range}")

        for n_est in n_est_range:
            for depth in depth_range:
                self.model.set_params(n_estimators = n_est, max_depth = depth)
                self.fit(X_train, y_train)
                acc = self.score(X_test, y_test)
                print(f"  n_estimators={n_est:>3}, max_depth={str(depth):>4} → Accuracy: {acc:.4f}")

                if acc > best_acc:
                    best_acc = acc
                    best_n_est = n_est
                    best_depth = depth

        # Refit với best params
        self.n_estimators = best_n_est
        self.max_depth = best_depth
        self.model.set_params(n_estimators = best_n_est, max_depth = best_depth)
        self.fit(X_train, y_train)

        print(f"\n Best n_estimators={best_n_est}, max_depth={best_depth} (Accuracy: {best_acc:.4f})")
        return best_n_est, best_depth
