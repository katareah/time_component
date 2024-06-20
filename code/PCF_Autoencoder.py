class RPCFGANTrainer(BaseTrainer):
    def __init__(self, G, D, train_dl, config, **kwargs):
        """
        Trainer class for PCFGAN with time series embedding,
            which provide extrac time series reconstruction functionality.

        Args:
            G (torch.nn.Module): RPCFG generator model.
            D (torch.nn.Module): RPCFG discriminator model (character function).
            train_dl (torch.utils.data.DataLoader): Training data loader.
            config: Configuration object containing hyperparameters and settings.
            **kwargs: Additional keyword arguments for the base trainer class.
        """
        super(RPCFGANTrainer, self).__init__(
            G=G,
            G_optimizer=torch.optim.Adam(
                G.parameters(), lr=config.lr_G, betas=(0, 0.9)
            ),
            **kwargs
        )
        self.config = config
        self.add_time = config.add_time
        self.train_dl = train_dl
        self.D_steps_per_G_step = config.D_steps_per_G_step
        self.D = D
        char_input_dim = self.config.D_out_dim

        self.D_optimizer = torch.optim.Adam(
            self.D.parameters(), lr=config.lr_D, betas=(0, 0.9)
        )
        self.char_func = char_func_path(
            num_samples=config.M_num_samples,
            hidden_size=config.M_hidden_dim,
            input_size=char_input_dim,
            add_time=self.add_time,
            init_range=config.init_range,
        )
        self.char_func1 = char_func_path(
            num_samples=config.M_num_samples,
            hidden_size=config.M_hidden_dim,
            input_size=char_input_dim,
            add_time=self.add_time,
            init_range=config.init_range,
        )

        self.char_optimizer = torch.optim.Adam(
            self.char_func.parameters(), betas=(0, 0.9), lr=config.lr_M
        )
        self.char_optimizer1 = torch.optim.Adam(
            self.char_func1.parameters(), betas=(0, 0.9), lr=config.lr_M
        )

        self.averaged_G = swa_utils.AveragedModel(G)
        self.G_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.G_optimizer, gamma=config.gamma
        )
        self.D_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.D_optimizer, gamma=config.gamma
        )
        self.M_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.char_optimizer, gamma=config.gamma
        )
        self.M_lr_scheduler1 = torch.optim.lr_scheduler.ExponentialLR(
            self.char_optimizer1, gamma=config.gamma
        )
        self.Lambda1 = self.config.Lambda1
        self.Lambda2 = self.config.Lambda2
        if self.config.BM:
            self.noise_scale = self.config.noise_scale
        else:
            self.noise_scale = 0.3

    def fit(self, device):
        """
        Trains the PCFGAN model.

        Args:
            device: Device to perform training on.
        """
        self.G.to(device)
        self.D.to(device)
        self.char_func.to(device)
        self.char_func1.to(device)

        for i in tqdm(range(self.n_gradient_steps)):
            self.step(device, i)
            if i > self.config.swa_step_start:
                self.averaged_G.update_parameters(self.G)

    def step(self, device, step):
        """
        Performs one training step.

        Args:
            device: Device to perform training on.
            step (int): Current training step.
        """
        for i in range(self.D_steps_per_G_step):
            # generate x_fake

            with torch.no_grad():
                z = (
                    self.noise_scale
                    * torch.randn(
                        self.config.batch_size,
                        self.config.n_lags,
                        self.config.G_input_dim,
                    )
                ).to(device)
                if self.config.BM:
                    z = z.cumsum(1)
                else:
                    pass
                x_real_batch = next(iter(self.train_dl))[0].to(device)
                x_fake = self.G(
                    batch_size=self.batch_size,
                    n_lags=self.config.n_lags,
                    z=z,
                    device=device,
                )

            self.M_trainstep(x_fake, x_real_batch, z)
            D_loss, enc_loss, reg_loss = self.D_trainstep(
                x_fake, x_real_batch, z, self.Lambda1, self.Lambda2
            )
            if i == 0:
                self.losses_history["D_loss"].append(D_loss)
                self.losses_history["recovery_loss"].append(enc_loss)
                self.losses_history["regularzation_loss"].append(reg_loss)

        G_loss = self.G_trainstep(x_real_batch, device, step)
        self.losses_history["G_loss"].append(G_loss)
        torch.cuda.empty_cache()
        if step % 500 == 0:
            self.D_lr_scheduler.step()
            self.G_lr_scheduler.step()
            for param_group in self.D_optimizer.param_groups:
                print("Learning Rate: {}".format(param_group["lr"]))
        else:
            pass

        if step % 2000 == 0:
            latent_x_fake = self.D(x_fake)
            latent_x_real = self.D(x_real_batch)
            x_real_dim = latent_x_fake.shape[-1]
            for j in range(x_real_dim):
                plt.plot(to_numpy(latent_x_fake[:100, :, j]).T, "C%s" % j, alpha=0.1)
            plt.savefig(
                pt.join(self.config.exp_dir, "latent_x_fake_" + str(step) + ".png")
            )
            plt.close()
            for j in range(x_real_dim):
                plt.plot(to_numpy(latent_x_real[:100, :, j]).T, "C%s" % j, alpha=0.1)
            plt.savefig(
                pt.join(self.config.exp_dir, "latent_x_real_" + str(step) + ".png")
            )
            plt.close()

    def G_trainstep(self, x_real, device, step):
        """
        Performs one training step for the generator.

        Args:
            x_real: Real samples for training.
            device: Device to perform training on.
            step (int): Current training step.

        Returns:
            float: Generator loss value.
        """
        x_fake = self.G(
            batch_size=self.batch_size, n_lags=self.config.n_lags, device=device
        )

        toggle_grad(self.G, True)
        self.G.train()
        self.G_optimizer.zero_grad()
        self.D.train()
        self.char_func.train()
        G_loss = self.char_func.distance_measure(self.D(x_real), self.D(x_fake))
        G_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.config.grad_clip)
        self.G_optimizer.step()
        if step % 2000 == 0:
            self.evaluate(x_fake, x_real, step, self.config)
        return G_loss.item()

    def M_trainstep(self, x_fake, x_real, z):
        """
        Performs one training step for the character function.

        Args:
            x_fake: Fake samples generated by the generator.
            x_real: Real samples for training.
            z: Latent noise used for generating fake samples.
        """
        toggle_grad(self.char_func, True)
        self.char_func.train()
        self.D.train()
        self.char_optimizer.zero_grad()
        char_loss = -self.char_func.distance_measure(self.D(x_real), self.D(x_fake))
        char_loss.backward()
        self.char_optimizer.step()
        toggle_grad(self.char_func, False)

        toggle_grad(self.char_func1, True)
        self.char_func1.train()
        self.char_optimizer1.zero_grad()
        char_loss1 = self.char_func1.distance_measure(self.D(x_real), z)
        char_loss1.backward()
        self.char_optimizer1.step()
        toggle_grad(self.char_func1, False)

    def D_trainstep(self, x_fake, x_real, z, Lambda1, Lambda2):
        """
        Performs one training step for the discriminator.

        Args:
            x_fake: Fake samples generated by the generator.
            x_real: Real samples for training.
            z: Latent noise used for generating fake samples.
            Lambda1: Weight for the reconstruction loss.
            Lambda2: Weight for the regularization loss.

        Returns:
            Tuple[float, float, float]: Discriminator loss, reconstruction loss, and regularization loss.
        """
        x_real.requires_grad_()
        self.D.train()

        self.char_func.train()
        self.char_func1.train()
        toggle_grad(self.D, True)
        rec_loss = Lambda1 * nn.MSELoss()(self.D(x_fake), z)
        reg_loss = Lambda2 * self.char_func1.distance_measure(self.D(x_real), z)
        g_loss = -self.char_func.distance_measure(self.D(x_real), self.D(x_fake))

        d_loss = g_loss + rec_loss + reg_loss
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.config.grad_clip)
        self.D_optimizer.step()
        toggle_grad(self.D, False)

        return g_loss.item(), rec_loss.item(), reg_loss.item()