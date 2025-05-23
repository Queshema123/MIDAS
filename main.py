from midas import MIDAS
from weights import BetaWeights, AlmonExpPolyWeights, AlmonPolynomialWeights, Weights as UMIDAS
from matplotlib import pyplot as plt

if __name__ == "__main__":
    folder_path = "data\\filtered_data\\stl" 
    model = MIDAS(folder_path, AlmonExpPolyWeights)
    model.train('2019-03-31', 'GDP', 3)
    print(f"Результаты обучения\nMAE: {model.train_results['MAE']}\nRMSE: {model.train_results['RMSE']}")
    plt.plot(model.train_results['dates'], model.train_results['vals'])
    plt.plot(model.train_results['dates'], model.train_results['forecast'])
    plt.legend()
    plt.show()
    input()
