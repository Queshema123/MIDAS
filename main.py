from midas import MIDAS
from weights import BetaWeights, AlmonExpPolyWeights, AlmonPolynomialWeights
from matplotlib import pyplot as plt

# Основной скрипт
if __name__ == "__main__":
    folder_path = "data\\filtered_data\\kalman" 
    model = MIDAS(folder_path, AlmonExpPolyWeights)
    model.train('2019-03-31', 'GDP')
    print(f"Результаты обучения\nMAE: {model.train_results['MAE']}\nRMSE: {model.train_results['RMSE']}")
    plt.plot(model.train_results['dates'], model.train_results['vals'])
    plt.plot(model.train_results['dates'], model.train_results['forecast'])
    plt.legend()
    plt.show()
    input()
