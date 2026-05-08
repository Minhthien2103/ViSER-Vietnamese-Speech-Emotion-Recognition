from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class KNNModel:
    """
    KNN classifier cho ViSER.
    - KNN nhạy cảm với scale → tích hợp StandardScaler.
    - Dùng distance weighting để neighbors gần có ảnh hưởng lớn hơn.
    """

    def __init__(self, n_neighbors = 5):
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.model = KNeighborsClassifier(
            n_neighbors = n_neighbors,
            weights = 'distance',
            metric = 'minkowski',
            n_jobs = -1,
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


    def find_best_k(self, X_train, y_train, X_test, y_test, k_range = range(3, 26, 2)):
        """
            Tìm k tốt nhất, cập nhật model, trả về best_k.
        """

        best_k, best_acc = self.n_neighbors, 0

        for k in k_range:
            self.n_neighbors = k
            self.model.set_params(n_neighbors = k)
            self.fit(X_train, y_train)
            acc = self.score(X_test, y_test)
            print(f"  k={k:2d} → Accuracy: {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_k = k

        # Refit với best_k
        self.n_neighbors = best_k
        self.model.set_params(n_neighbors = best_k)
        self.fit(X_train, y_train)

        print(f"\n Best k = {best_k} (Accuracy: {best_acc:.4f})")
        return best_k
