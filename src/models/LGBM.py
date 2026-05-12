from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

class LGBMModel:
    """
    LightGBM classifier cho ViSER.
    - Cực kỳ nhanh, ít tốn bộ nhớ.
    """

    def __init__(self, n_estimators=100, max_depth=-1, learning_rate=0.1, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1 # Tắt các cảnh báo (warning) không cần thiết khi train
        )


    def fit(self, X, y):
        self.model.fit(X, y)
        return self


    def predict(self, X):
        return self.model.predict(X)


    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


    def find_best_params(self, X_train, y_train, X_test, y_test):
        """
        Grid search trên max_depth và learning_rate.
        Lưu ý: max_depth=-1 nghĩa là không giới hạn độ sâu.
        """
        depth_range = [-1, 5, 10]
        lr_range = [0.01, 0.1, 0.2]

        best_acc = 0
        best_depth, best_lr = self.max_depth, self.learning_rate

        print(f"  Grid search LightGBM: max_depth={depth_range}, learning_rate={lr_range}")

        for depth in depth_range:
            for lr in lr_range:
                self.model.set_params(max_depth=depth, learning_rate=lr)
                
                self.fit(X_train, y_train)
                acc = self.score(X_test, y_test)
                
                print(f"  max_depth={depth:>2}, learning_rate={lr:>4} → Accuracy: {acc:.4f}")

                if acc > best_acc:
                    best_acc = acc
                    best_depth = depth
                    best_lr = lr

        # Refit với best params
        self.max_depth = best_depth
        self.learning_rate = best_lr
        self.model.set_params(max_depth=best_depth, learning_rate=best_lr)
        self.fit(X_train, y_train)

        print(f"\n Best max_depth={best_depth}, learning_rate={best_lr} (Accuracy: {best_acc:.4f})")
        return best_depth, best_lr