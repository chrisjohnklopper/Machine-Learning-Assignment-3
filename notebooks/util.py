from matplotlib import pyplot as plt

def describe(data):
    description = data.describe().T
    description.rename(columns={'count': 'Count',
                                'mean': 'Mean',
                                '25%': '1st Quartile',
                                '50%': 'Median',
                                '75%': '3rd Quartile',
                                'max': 'Max',
                                'std': 'Standard Deviation',
                               }, inplace=True)
    return description.round(2)

def boxplot(data, feature):
    plt.boxplot(data[feature], vert=False)
    plt.xlabel(feature)
    plt.show()
