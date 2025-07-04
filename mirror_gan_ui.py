import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.decomposition import PCA
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class ProductiveGANTrainer(QtCore.QObject):
    update_plot = QtCore.pyqtSignal(float, float, float, int, object, object)

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.G = Generator().to(self.device)
        self.D = Discriminator().to(self.device)
        self.loss_fn = nn.BCELoss()
        self.g_opt = optim.Adam(self.G.parameters(), lr=0.005)
        self.d_opt = optim.Adam(self.D.parameters(), lr=0.005)
        self.epoch = 0
        self.generated_log = []  # for meme outputs
        self.drift_history = []  # for drift visualization

    def real_data(self, n):
        return torch.sin(torch.linspace(-np.pi, np.pi, n)).unsqueeze(1)

    def noise(self, n):
        return torch.randn(n, 5)

    def public_signal(self, fake_data):
        fatigue_factor = max(0.2, 1.0 - self.epoch / 500.0)
        reward = 1 - torch.abs(fake_data).mean()
        return fatigue_factor * reward

    def step(self):
        self.G.train()
        self.D.train()

        real = self.real_data(64)
        noise = self.noise(64)
        fake = self.G(noise).detach()

        # Discriminator training
        d_real = self.D(real)
        d_fake = self.D(fake)
        d_loss = self.loss_fn(d_real, torch.ones_like(d_real)) + self.loss_fn(d_fake, torch.zeros_like(d_fake))

        self.d_opt.zero_grad()
        d_loss.backward()
        self.d_opt.step()

        # Generator training
        noise = self.noise(64)
        generated = self.G(noise)
        d_pred = self.D(generated)
        g_loss = self.loss_fn(d_pred, torch.ones_like(d_pred))

        # Public feedback
        p_reward = self.public_signal(generated)
        g_total_loss = g_loss - 0.5 * p_reward

        self.g_opt.zero_grad()
        g_total_loss.backward()
        self.g_opt.step()

        if self.epoch % 10 == 0:
            self.generated_log.append(generated.detach().cpu().numpy())
            self.drift_history.append(noise.detach().cpu().numpy())

        self.epoch += 1
        self.update_plot.emit(
            g_loss.item(), d_loss.item(), p_reward.item(), self.epoch,
            generated.detach().cpu().numpy(), noise.detach().cpu().numpy()
        )

class ProductiveGANVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Productive Conflict GAN with Drift & Fatigue")
        self.resize(1000, 600)

        main_widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        self.plot_losses = pg.PlotWidget(title="Losses & Public Signal")
        self.plot_losses.addLegend()
        self.gen_curve = self.plot_losses.plot(pen='r', name="Generator Loss")
        self.disc_curve = self.plot_losses.plot(pen='b', name="Discriminator Loss")
        self.pub_curve = self.plot_losses.plot(pen='g', name="Public Reward", symbol='o', symbolSize=5)

        self.plot_drift = pg.PlotWidget(title="Idea Drift (PCA of Latents)")
        self.plot_drift.setXRange(-3, 3)
        self.plot_drift.setYRange(-3, 3)
        self.drift_scatter = self.plot_drift.plot(pen=None, symbol='x', symbolBrush='y')

        layout.addWidget(self.plot_losses, 2)
        layout.addWidget(self.plot_drift, 1)

        self.g_losses, self.d_losses, self.p_rewards, self.epochs = [], [], [], []

        self.trainer = ProductiveGANTrainer()
        self.trainer.update_plot.connect(self.update_plot)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.trainer.step)
        self.timer.start(100)

    def update_plot(self, g_loss, d_loss, p_reward, epoch, memes, latents):
        self.epochs.append(epoch)
        self.g_losses.append(g_loss)
        self.d_losses.append(d_loss)
        self.p_rewards.append(p_reward)

        self.gen_curve.setData(self.epochs, self.g_losses)
        self.disc_curve.setData(self.epochs, self.d_losses)
        self.pub_curve.setData(self.epochs, self.p_rewards)

        # Idea drift
        if len(self.trainer.drift_history) > 1:
            all_latents = np.vstack(self.trainer.drift_history)
            pca = PCA(n_components=2)
            drift_coords = pca.fit_transform(all_latents)
            self.drift_scatter.setData(drift_coords[:, 0], drift_coords[:, 1])

def run_app():
    app = QtWidgets.QApplication(sys.argv)
    window = ProductiveGANVisualizer()
    window.show()
    sys.exit(app.exec_())

run_app()
