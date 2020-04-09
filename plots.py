import matplotlib.pyplot as plt


class Plots(object):
    size = 10

    def correlation(self, corr):
        fig, ax = plt.subplots(figsize=(self.size, self.size))
        ax.matshow(corr)
        plt.xticks(range(len(corr.columns)), corr.columns)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.show()
