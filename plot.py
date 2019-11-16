from visdom import Visdom
import numpy as np

class VisdomLinePlotter:
    """Plots to Visdom"""

    def __init__(self, env_name='main', port=8080, disable=False):
        self.disable = disable
        self.env = env_name
        self.plots = {}

        if not self.disable:
            try:
                self.viz = Visdom(port=port)
            except (ConnectionError, ConnectionRefusedError) as e:
                raise ConnectionError(
                    "Visdom Server not running, please launch it with `visdom` in the terminal")


    def clear(self):
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y, xlabel='epochs'):
        if self.disable:
            return

        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel=xlabel,
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update='append')