import torch
import torch.nn as nn
from torch.optim import Adam

from tensorboardX import SummaryWriter

from tqdm.auto import tqdm

class inpcrd_DDPM_cond(nn.Module):
    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None):
        super(inpcrd_DDPM_cond, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, label, eta=None):
        n, atsdims=x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, atsdims).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1) * eta 

        return noisy

    def backward(self, x, t, label):
        return self.network(x, t, label)

def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding

class inpcrd_Seq_Conditioned(nn.Module): 
    def __init__(self, max_size, n_steps=1000, time_emb_dim=100, Natoms=71, dims=3, NClasses=2, label_emb_dim=100): 
        super(inpcrd_Seq_Conditioned, self).__init__() 
        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim) 
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim) 
        self.time_embed.requires_grad_(False) 
        #
        self.label_embedding = nn.Embedding(NClasses, label_emb_dim)
        #
        N1=Natoms*dims
        N2=10*N1
        #                                                                                        
        self.te_N1 = self._make_te(time_emb_dim, N1) 
        self.te_N2 = self._make_te(time_emb_dim, N2) 
        self.te_N2a = self._make_te(time_emb_dim, N2) 
        #
        self.le_N1 = self._make_te(label_emb_dim, N1) 
        self.le_N2 = self._make_te(label_emb_dim, N2) 
        self.le_N2a = self._make_te(label_emb_dim, N2)
        #
        self.lin_N1N2=nn.Linear(N1,N2)
        self.lin_N2N2=nn.Linear(N2,N2)
        self.lin_N2N1=nn.Linear(N2,N1)
        #
        self.LReLU_a=nn.LeakyReLU(0.2)
        self.LReLU_b=nn.LeakyReLU(0.2)

    def forward(self, x, t, labels):                                                                     
        c = self.label_embedding(labels)
        t = self.time_embed(t)
        n = len(x)

        # N1 -> N2
        out=x
        out=out + self.te_N1(t).reshape(n, -1) + self.le_N1(c).reshape(n, -1)
        out=self.lin_N1N2(out) 
        out=self.LReLU_a(out)
        #
        # N2 -> N2
        out=out + self.te_N2(t).reshape(n, -1) + self.le_N2(c).reshape(n, -1)
        out=self.lin_N2N2(out) 
        out=self.LReLU_b(out)
        # N2 -> N1
        out=out + self.te_N2a(t).reshape(n, -1) + self.le_N2a(c).reshape(n, -1)
        out=self.lin_N2N1(out)
        return out

    def _make_te(self, dim_in, dim_out): 
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )


def training_loop(ddpm, loader, n_epochs, optim, device, output_directory, store_path="ddpm_model.pt"):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps

    summary_writer = SummaryWriter(output_directory) # Usage: tensorboard --logdir=./
    color1="#5171A5"
    color2="#2E4052"
    first_loss_determined=False
    n_params=0
    for param in ddpm.parameters():
        n_params+=1
    print(f"{n_params=}")
    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour=color1):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour=color2)):
            # Loading data
            x0 = batch[0].to(device)
            lab = batch[1].to(device)
            n = len(x0)

            # Picking some noise for each of the structures in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device) 
            t = torch.randint(0, n_steps, (n,)).to(device)

            # Computing the noisy structures based on x0 and the time-step (forward process)
            noisy_strcs = ddpm(x0, t, lab, eta)
            
            # Getting model estimation of noise based on the structures and the time-step
            eta_theta = ddpm.backward(noisy_strcs, t.reshape(n, -1), lab)
            
            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(eta_theta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)
            summary_writer.add_scalar('Loss',torch.FloatTensor([epoch_loss]),global_step=epoch)

        log_string = f"\nLoss at epoch {epoch + 1}: {epoch_loss:.3f}"

        if first_loss_determined==False:
            first_loss=epoch_loss
            first_loss_determined=True

        training_wrong=False
        if epoch_loss > 10*first_loss:
            log_string += " --> Training gone wrong. Model will be re-loaded"
            training_wrong=True
            ddpm.load_state_dict(torch.load(store_path, map_location=device)) 
            
        # Storing the model
        if not training_wrong:
            if best_loss > epoch_loss:
                best_loss = epoch_loss
                torch.save(ddpm.state_dict(), store_path)
                log_string += " --> Best model ever (stored)"

        print(log_string)

