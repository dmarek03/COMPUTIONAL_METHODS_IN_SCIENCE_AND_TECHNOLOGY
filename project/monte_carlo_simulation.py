import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def get_data(stocks, period):
    stock_data =  yf.download(stocks, period=period, auto_adjust=True)
    stock_data = stock_data['Close']
    returns =  stock_data.pct_change()
    mean_returns =  returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix


def get_weights(size:int):
    weights = np.random.random(size)
    weights /= np.sum(weights)
    return weights


def monte_carlo_simulation(simulation_number:int, period_of_simulation:int,number_of_stocks,mean_return, cov_matrix, initial_portfolio:float):
    mean_m = np.full(shape=(period_of_simulation, number_of_stocks), fill_value=mean_return)
    mean_m = mean_m.T

    portfolio_sims = np.full(shape=(period_of_simulation, simulation_number), fill_value=0.0)
    for m in range(simulation_number):
        weights = get_weights(number_of_stocks)
        Z = np.random.normal(size=(period_of_simulation, len(weights)))
        L = np.linalg.cholesky(cov_matrix)

        daily_return = mean_m + np.inner(L, Z)

        portfolio_sims[:, m] = np.cumprod(np.inner(weights, daily_return.T) + 1) * initial_portfolio

    return portfolio_sims


def draw_plot(data):
    plt.plot(data)
    plt.ylabel("Portfolio Value ($)")
    plt.xlabel('Days')
    plt.title('MC simulation of a stock portfolio')
    plt.show()


def mcVar(returns, alpha=1):
    print(f'{returns=}')
    return np.percentile(returns, alpha)


def mcCVAr(returns, alpha=1):
    below_var = returns <= mcVar(returns, alpha=alpha)
    return returns[below_var].mean()



def main() -> None:

    stock_list = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'META']

    period = '1y'
    simulation_number = 10000
    period_of_simulation =  365
    initial_portfolio_value = 10000.00
    data_mean, data_cov_matrix = get_data(stock_list, period)
    number_of_stocks = len(stock_list)

    portfolio_simulations =  monte_carlo_simulation(
        simulation_number,
        period_of_simulation,
        number_of_stocks,
        data_mean,
        data_cov_matrix,
        initial_portfolio_value
    )
    port_result = pd.Series(portfolio_simulations[-1, :])

    VaR = initial_portfolio_value - mcVar(port_result, alpha=5)
    CVaR = initial_portfolio_value - mcCVAr(port_result, alpha=5)

    print(f'VaR ${round(VaR, 2)}')
    print(f'CVaR ${round(CVaR, 2)}')
    print(f'Mean portfolio result {round(portfolio_simulations.mean(), 2)}')
    draw_plot(portfolio_simulations)


if __name__ == '__main__':
    main()
