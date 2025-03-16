'''
This is Homework 1 in COMP4220-Machine Learning 
University of Massachusetts Lowell
'''
import numpy as np
import matplotlib.pyplot as plt
import itertools, functools
np.random.seed(1234)  # for reproducibility


def generate_synthetic_data(func, sample_size, std):
    '''Generates 1D synthetic data for regression
    Inputs:
        func: a function representing the curve to sample from
        sample_size: number of points to generate
        std: standrad deviation for additional noise
    '''
    x = np.linspace(0, 1, sample_size)
    t = func(x) + np.random.normal(scale=std, size=x.shape)
    return x, t


def func(x):
      # part (a)
    return np.sin(2 * np.pi * x)


class PolynomialFeature:
    '''Class for generating and transforming polynomial features'''
    
    def __init__(self, degree=2):
        assert isinstance(degree, int)
        self.degree = degree

    def transform(self, x):
        if x.ndim == 1:
            x = x[:, None]  #ensure size (:,1) instead of (:,)
        x_t = x.transpose() #return the data to its initial shape
        features = [np.ones(len(x))]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(x_t, degree):
                features.append(functools.reduce(lambda x, y: x * y, items))
        return np.asarray(features).transpose()


class Regression:
    '''Basic class for the regression algorithms'''
    
    pass



class LinearRegression(Regression):
    def __init__(self):
        self.w = None

    def _reset(self):
        self.w = None

    def _fit(self, x_train: np.ndarray, y_train: np.ndarray):
        # This is the fit function. Implement the equations that calculate 
        # the mean and variance of the soution 
        # part (e)
        self.w = np.linalg.pinv(x_train) @ y_train

    def _predict(self, x: np.ndarray, return_std: bool=False):
        # part (f)
        return x @ self.w


def rmse(a, b):
    '''Calculates the RMSE error between two vectors
    '''
    # part (j)
    return np.sqrt(np.mean((a - b) ** 2))


def main():
    # part (b) - generate training set
    x_train, t_train = generate_synthetic_data(func, 10, 0.25)
     # part (c) - generate test set
    x_test, t_test = generate_synthetic_data(func, 100, 0)

    # part (d)
    degrees = [0, 1, 3, 9]
    errors_train, errors_test = [], []

    for i, degree in enumerate(degrees):
        poly = PolynomialFeature(degree)
        X_train = poly.transform(x_train)
        X_test = poly.transform(x_test)
        # part (g)
        model = LinearRegression()
        model._fit(X_train, t_train)
        # part (h)
        y_train_pred = model._predict(X_train)
        y_test_pred = model._predict(X_test)
        
        errors_train.append(rmse(t_train, y_train_pred))
        errors_test.append(rmse(t_test, y_test_pred))
        # part (i)
        plt.subplot(2, 2, i + 1)
        plt.scatter(x_train, t_train, color='blue', label='Training Data')
        plt.plot(x_test, func(x_test), linestyle='dashed')
        plt.plot(x_test, y_test_pred, label=f'Order {degree}')
        plt.legend()
        plt.title(f'Degree {degree}')
    plt.tight_layout()
    plt.show()
    # part (k)
    degrees = [0, 3, 6, 9]
    plt.plot(degrees, errors_train, marker='o', label='Train RMSE')
    plt.plot(degrees, errors_test, marker='s', label='Test RMSE')
    plt.xlabel('Polynomial Order')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.xticks(degrees)
    plt.show()

    # part (l)
    print("Table of trained weights (w*):")
    for degree in range(10):
        poly = PolynomialFeature(degree)
        X_train = poly.transform(x_train)
        model = LinearRegression()
        model._fit(X_train, t_train)
        print(f'M={degree}:', model.w)


if __name__ == '__main__':
    print('--- Homework 1 ---')
    main()
