from midas import MIDAS
from weights import BetaWeights, AlmonExpPolyWeights, AlmonPolynomialWeights, Weights as UMIDAS
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    folder_path = "data\\filtered_data\\stl" 
    model = MIDAS(folder_path, AlmonExpPolyWeights, lambda_l1=0.2, lambda_l2=0.1)
    #model.train('2019-03-31', 'GDP', 6)
    model.window_train('2019-03-31', 'GDP', 3, 8, 4)
    print(f"Результаты обучения\nMAE: {model.train_results['MAE']}\nRMSE: {model.train_results['RMSE']}")
    y_true = np.array(model.train_results['vals'])
    y_pred = np.array(model.train_results['forecast'])
    ape = np.abs((y_pred - y_true) / y_true) * 100.0
    mape = np.mean(ape)
    print(f"Относительная ошибка составляет {mape:.2f}%")
    plt.plot(model.train_results['dates'], model.train_results['vals'])
    plt.plot(model.train_results['dates'], model.train_results['forecast'])
    plt.legend()
    plt.show()
    input()

# l1 = 0.2, l2 = 0.1