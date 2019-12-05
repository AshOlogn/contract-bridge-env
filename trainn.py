def __init__(self, i_h = 144, i_w = 256, n_a = 7):
        super(DeepQNetwork, self).__init__()
        self.fx = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        def out_comp(ins, ks, s):
            return (ins - (ks - 1) - 1) // s + 1
        
        def adj_size(ins, convs, ks = 5, s = 2):
            sz = ins
            for _ in range(convs):
                sz = out_comp(sz, ks, s)
            return sz

        l_i = adj_size(i_h, 4) * adj_size(i_w, 4) * 64 + 9

        self.cl = nn.Sequential(
            nn.Linear(l_i, 2048),
            nn.ReLU(),
            nn.Linear(2048, 64),
            nn.ReLU(),
            nn.Linear(64, n_a)
        )

    def forward(self, x, gps_md):
        # x contains three 3 x 144 x 256 image
        x = self.fx(x)
        ifx = x.view(x.shape[0], -1)
        bsz = gps_md.shape[0]
        gps_md = gps_md.unsqueeze(0).permute(1,0,2)
        x = torch.cat([ifx, gps_md[:,:,0], gps_md[:,:,1], gps_md[:,:,2], gps_md[:,:,3], gps_md[:,:,4], gps_md[:,:,5], gps_md[:,:,6], gps_md[:,:,7], gps_md[:,:,8]], dim=1)
        return self.cl(x)


def env_reset(client):
    print("Resetting Environment...")
    client.reset()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()

def train(args):
    batch_size = 64
    gamma = 0.999
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200
    target_update = 10 # update target network after 10 episodes
    steps_done = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # networks
    policy_network = DeepQNetwork().to(device)
    target_network = DeepQNetwork().to(device)
    target_network.load_state_dict(policy_network.state_dict())

    # optimizer + loss func + replay memory
    optimizer = optim.RMSprop(policy_network.parameters())
    criterion = nn.SmoothL1Loss()
    replay_memory = ReplayMemory(5000)

    # connect to the AirSim simulator 
    #rotation_duration = 1

    lat_off = 0.001
    lng_off = 0.001
    alt_off = 0

    # get gps data for drone at current timestep
    gps_data = client.getGpsData()
    geo_data = gps_data.gnss.geo_point
    vel_data = gps_data.gnss.velocity


    for episode in range(int(args.episodes)):

        # set policy and target networks to train and eval respectively
        policy_network.train()
        target_network.eval()

        # reset environment at the start of every episode
        env_reset(client)

        last_state = get_state(client, device)
        current_state = get_state(client, device)

        state = current_state - last_state

        gps_md = get_gps_md(client, dest_alt, dest_lat, dest_lng)
        gps_md_no_tensor = gps_md
        gps_md = torch.tensor(gps_md, device=device).unsqueeze(0)
        #current_rotation_angle = 0
        start_lat = gps_md_no_tensor[1]
        start_lng = gps_md_no_tensor[2]

        slope = [(dest_lng - start_lng) / (dest_lat - start_lat), (dest_lat - start_lat) / (dest_lng - start_lng)]
        intercept = [dest_lng - slope[0] * dest_lat, dest_lat - slope[1] * dest_lng]

        total_dist = math.sqrt((dest_lng - start_lng) ** 2 + (dest_lat - start_lat) ** 2)

        print("Current episode destination: alt: {}, lat: {}, lng: {}".format(dest_alt, dest_lat, dest_lng))

        for t in count():
            # select an action using the policy network
            sample = random.random() # decide if we will follow the policy or explore
            eps_threshold = eps_end + (eps_start - eps_end) * \
                math.exp(-1. * steps_done / eps_decay)
            steps_done += 1
            # follow policy if above threshold otherwise do random action
            # (encourages exploration early on and then favors policy)
            action = torch.tensor([[random.randrange(num_actions)]], device=device, dtype=torch.long)
            if sample > eps_threshold:
                with torch.no_grad():
                    action = policy_network(state, gps_md).max(1)[1].view(1, 1)

            # rotate so image data is being recieved correctly from frontal view            
            #rotation_angle = get_action_rotation_angle(action)
            #in_case_no_movement_rotation_angle = current_rotation_angle
            #if (rotation_angle is not None) and (current_rotation_angle != rotation_angle):
            #    client.rotateByYawRateAsync(-current_rotation_angle, rotation_duration).join()
            #    client.rotateByYawRateAsync(rotation_angle, rotation_duration).join()
            #    current_rotation_angle = rotation_angle

            # take predicted action
            quad_offset = interpret_action(action) 
            quad_vel = client.getMultirotorState().kinematics_estimated.linear_velocity
            client.moveByVelocityAsync(quad_vel.x_val+quad_offset[0], quad_vel.y_val+quad_offset[1], quad_vel.z_val+quad_offset[2], 2, drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join()
            
            quad_state = client.getMultirotorState().kinematics_estimated.position
            quad_vel = client.getMultirotorState().kinematics_estimated.linear_velocity
            collision_info = client.simGetCollisionInfo()

            last_state = current_state
            current_state = get_state(client, device)
            next_state = current_state - last_state

            next_gps_md = get_gps_md(client, dest_alt, dest_lat, dest_lng)
            next_gps_md_no_tensor = next_gps_md
            next_gps_md = torch.tensor(next_gps_md, device=device).unsqueeze(0)

            reward, done, no_movement, arrived_dest = compute_reward(quad_state, quad_vel, collision_info, gps_md_no_tensor, next_gps_md_no_tensor, slope, intercept, total_dist)
            reward = torch.tensor([reward], device=device)

            #if no_movement:
            #    current_rotation_angle = in_case_no_movement_rotation_angle

            if done or no_movement:
                next_state = None
                next_gps_md = None
                next_gps_md_no_tensor = None
            
            replay_memory.push(state, action, next_state, reward, gps_md, next_gps_md)
            action_strings = ["hover", "straight", "right", "down", "back", "left", "up"]
            print("Action taken: {}\tReward Gained: {}\tReplay Memory Size: {}".format(
                action_strings[int(action.item())], reward.item(), len(replay_memory)
            ))

            state = next_state
            gps_md = next_gps_md
            gps_md_no_tensor = next_gps_md_no_tensor

            if len(replay_memory) >= batch_size:
                transitions = replay_memory.sample(batch_size)
                batch = Transition(*zip(*transitions))
                mask = torch.tensor(tuple(map(lambda s: s is not None, 
                                    batch.next_state)), device=device, dtype=torch.uint8)
                
                next_states = torch.cat([s for s in batch.next_state if s is not None])
                next_gps_states = torch.cat([s for s in batch.next_gps_state if s is not None])
                state_batch = torch.cat(batch.state)
                gps_state_batch = torch.cat(batch.gps_state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)

                # computing Q(s_t, a)
                q_values = policy_network(state_batch, gps_state_batch).gather(1, action_batch)

                # computing V(s_{t+1})
                next_state_values = torch.zeros(batch_size, device=device)
                next_state_values[mask] = target_network(next_states, next_gps_states).max(1)[0].detach()
                expected_q_values = (next_state_values * gamma) + reward_batch

                # compute loss
                loss = criterion(q_values, expected_q_values.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                for param in policy_network.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

            if arrived_dest:
                dest_lat = start_lat + random.uniform(0.001, 0.002)
                dest_lng = start_lng + random.uniform(0.001, 0.002)

            if done or no_movement:
                break
        
        print("Episode {} completed".format(episode))
        # update the target network based on the current policy
        # also save the current policy network
        if episode % target_update == 0:
            target_network.load_state_dict(policy_network.state_dict())
            torch.save(policy_network.state_dict(), "models/policy-network-{}.pth".format(episode))