model = modules.CompILE(
    state_dim=args.state_dim,
    action_dim=args.action_dim,
    hidden_dim=args.hidden_dim,
    latent_dim=args.latent_dim,
    max_num_segments=args.num_segments,
    latent_dist=args.latent_dist,
    device=device).to(device)


parameter_list = list(model.parameters()) + sum([list(subpolicy.parameters()) for subpolicy in model.subpolicies], [])

optimizer = torch.optim.Adam(parameter_list, lr=args.learning_rate)

data_states = np.load(data_path + '_states.npy', allow_pickle=True)
data_actions = np.load(data_path + '_actions.npy', allow_pickle=True)

train_test_split = np.random.permutation(len(data_states))
train_test_split_ratio = 0.01

train_data_states = data_states[train_test_split[int(len(data_states)*train_test_split_ratio):]]
train_action_states = data_actions[train_test_split[int(len(data_states)*train_test_split_ratio):]]

test_data_states = data_states[train_test_split[:int(len(data_states)*train_test_split_ratio)]]
test_action_states = data_actions[train_test_split[:int(len(data_states)*train_test_split_ratio)]]

test_lengths = torch.tensor([len(state) for state in test_data_states]).to(device)
test_inputs = (torch.tensor(test_data_states).to(device), torch.tensor(test_action_states).to(device))

all_data_states = torch.tensor(data_states).to(device)
all_action_states = torch.tensor(data_actions).to(device)


all_inputs = (all_data_states, all_action_states)
perm = utils.PermManager(len(train_data_states), args.batch_size)


step = 0
rec = None
batch_loss = 0
batch_acc = 0
best_rec_acc = 0
best_nll = np.inf

if args.train_model:
    while step < args.iterations:
        optimizer.zero_grad()

        # Generate data.
        batch = perm.get_indices()
        batch_states, batch_actions = train_data_states[batch], train_action_states[batch]
        lengths = torch.tensor([len(state) for state in batch_states]).to(device)
        inputs = (torch.tensor(batch_states).to(device), torch.tensor(batch_actions).to(device))

        # Run forward pass.
        model.train()
        outputs = model.forward(inputs, lengths)
        loss, nll, kl_z, kl_b = utils.get_losses(inputs, outputs, args, beta_b=args.beta_b, beta_z=args.beta_z, prior_rate=args.prior_rate)

        loss.backward()
        optimizer.step()

        if step % args.log_interval == 5:
            model.eval()
            outputs = model.forward(test_inputs, test_lengths)
            acc, rec = utils.get_reconstruction_accuracy(test_inputs, outputs, args)

            batch_acc = acc.item()
            batch_loss = nll.item()
        
        step += 1