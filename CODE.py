
import os, time, pickle, cv2
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mimoEnv

#prep.....
#a.download
print(sys.executable) #check envir

git clone https://github.com/babybench/BabyBench2025_Starter_Kit.git

cd BabyBench2025_Starter_Kit

pip install -r requirements.txt

cd /Users/weiqili/PycharmProjects/Baby/BabyBench2025_Starter_Kit/MIMo
pip install -e .

#b.log test
import pickle

file_path = "/Users/weiqili/PycharmProjects/Baby/BabyBench2025_Starter_Kit/results/test_installation/logs/training.pkl"

with open(file_path, 'rb') as f:
    training_log = pickle.load(f)

print(training_log)

#enviro
import gymnasium as gym
import mimoEnv

env = gym.make("BabyBench")
obs, _ = env.reset()
print(obs)

#avoid negative step length
def to_torch_contig(x, device, dtype=torch.float32, add_batch=True):
    t = torch.from_numpy(np.ascontiguousarray(x)).to(device=device, dtype=dtype)
    return t.unsqueeze(0) if add_batch else t
#ensure directory exist
def ensure_dir(p):
    d = os.path.dirname(p) or "."
    os.makedirs(d, exist_ok=True)

#encoder
class Encoder(nn.Module):
    def __init__(self, proprio_dim, tactile_dim, visual_shape, out_dim=256):
        super().__init__()
        C, H, W = visual_shape
        self.visual = nn.Sequential(
            nn.Conv2d(C, 32, 5, 2, 2), nn.ReLU(),#Increase the number of channels for c (eye input) to 32
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),#Increase the number of channels for c (eye input) to 64
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128), nn.ReLU()#Map the aggregated 64dimensional visual global vector to a 128dimensional visual embedding.
        )
        self.proprio = nn.Linear(proprio_dim, 64)#Compress the original proprioceptive vector
        self.tactile = nn.Linear(tactile_dim, 64)#Compress the original tactile vector
        self.fuse = nn.Sequential(nn.Linear(128+64+64, out_dim), nn.ReLU())#visual+proprioceptive+tactile
    def forward(self, obs):
        v = self.visual(obs["rgb"] / 255.0)
        p = torch.relu(self.proprio(obs["proprio"]))
        t = torch.relu(self.tactile(obs["tactile"]))
        return self.fuse(torch.cat([v, p, t], dim=-1))

#policy value network construction (just a frame)
class PolicyValue(nn.Module):
    def __init__(self, in_dim, act_dim):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(in_dim,256), nn.ReLU(),#policy network,actor
                                nn.Linear(256, act_dim))
        self.v  = nn.Sequential(nn.Linear(in_dim,256), nn.ReLU(),#value network,critic
                                nn.Linear(256, 1))
    def forward(self, z):
        mu = self.pi(z)
        v  = self.v(z).squeeze(-1)
        return mu, v

#random network distillation
#I used RND to simulate an infant's curiosity about the external world.
#The stronger its curiosity and the more unexpected the outcome, the greater the reward—and vice versa.
class RND(nn.Module):
    def __init__(self, in_dim, hid=256):
        super().__init__()
        self.target = nn.Sequential(nn.Linear(in_dim,hid), nn.ReLU(),
                                    nn.Linear(hid,hid))
        for p in self.target.parameters():
            p.requires_grad = False#Fix the parameters
        self.predictor = nn.Sequential(
            nn.Linear(in_dim,hid), nn.ReLU(),
            nn.Dropout(0.1),#prevend overfitting
            nn.Linear(hid,hid)
        )
    def intrinsic_reward(self, z):#z=visual+proprioceptive+tactile from encoder
        with torch.no_grad(): #Disable gradients to ensure they do not backpropagate to the target network.
            t = self.target(z) #Train the network to predict the output of the target.
        p = self.predictor(z)
        return (p - t).pow(2).mean(dim=-1)  #predict->target

#becasue the tactile input was always to 0 in last training, i add tactile input maintain to stable the training
class TouchStats:
    def __init__(self, dim, beta=0.01):
        self.mu = np.zeros(dim, np.float32)# Initialize mean
        self.sigma = np.ones(dim, np.float32)#Initialize sd
        self.beta = beta
    def update(self, x):
        x = x.astype(np.float32)
        d = x - self.mu #current input-mean
        self.mu += self.beta * d
        self.sigma = np.sqrt((1 - self.beta) * (self.sigma**2) + self.beta * (x - self.mu)**2 + 1e-6)#Continuously updating tactile wave range

#Preserve the tactile of the previous frame
class TouchMemory:
    def __init__(self):
        self.prev_energy = None

touch_mem = TouchMemory()

#this function is design for making MIMo more likely to receive rewards for touch events, thereby learning self-touch.
def touch_bonus(x, stats: TouchStats, k=3.0, dead=0.1, z_clip=5.0, peak_thresh=2.0, use_delta=True):
    x = x.astype(np.float32)
    z = (x - stats.mu) / (stats.sigma + 1e-6)
    if z_clip is not None:
        z = np.clip(z, -z_clip, z_clip) #Avoid sudden spikes in a tactile channel that could cause high reward
    rms = float(np.sqrt((z ** 2).mean()))
    rms_part = np.tanh(max(0.0, rms - dead) / k) #basic reward,if touch,get reward
    max_abs = float(np.max(np.abs(z))) if z.size > 0 else 0.0#Check if any tactile channel exceeds the set threshold
    peak_part = 1.0 if max_abs >= peak_thresh else 0.0#Once the touch intensity reaches a peak, immediately award the full reward.
    delta_part = 0.0
    if use_delta: #Reward agents for proactively creating new touch points
        energy = float(np.linalg.norm(x))
        if touch_mem.prev_energy is not None:
            inc = max(0.0, energy - touch_mem.prev_energy)
            delta_part = np.tanh(inc / (np.sqrt(x.size) + 1e-6))
        touch_mem.prev_energy = energy
    w_peak, w_rms, w_delta = 0.5, 0.4, 0.1 #Weighted combination of the three,rms_part + peak_part + delta_part
    r = w_peak * peak_part + w_rms * rms_part + w_delta * delta_part
    return float(np.clip(r, 0.0, 1.0))

#this function is used for hand regard 
class RegardHelper:
    def __init__(self, H=64, W=64, crop=0.5):#size of the eye and percentage of the central area
        ch = int(H*crop/2); cw = int(W*crop/2)
        self.win = (H//2 - ch, H//2 + ch, W//2 - cw, W//2 + cw)#Pupil
        self.prev = None
    def motion_center_adv(self, eye_rgb):
        gray = cv2.cvtColor(eye_rgb, cv2.COLOR_RGB2GRAY)
        if self.prev is None:
            self.prev = gray; return 0.0
        diff = cv2.absdiff(gray, self.prev)
        self.prev = gray
        r0,r1,c0,c1 = self.win
        m_center = float(diff[r0:r1, c0:c1].mean())#Movement intensity in the central area （pupil）
        m_global = float(diff.mean()) + 1e-6 #Average motion intensity across the entire frame
        return float(np.tanh((m_center / m_global) - 1.0))#If the central area shows stronger movement than the overall field,
                                                        # it indicates that the object the baby is focusing on is located in the center

#Since the baby didn't stop when looking at his own hands in the video, I added a reward for when his eyes see his hands and they are actually moving.
# This is quite realistic, as the baby hasn't yet developed the concept of subject and object， he should be more interested in moving things.
def estimate_wrist_speed_from_obs(proprio_vec):#give baby extra rewards，encourage baby to wave his hands
    v = np.asarray(proprio_vec, np.float32)
    seg = v[-30:] if v.size >= 30 else v
    return float(np.linalg.norm(seg))

def hand_regard_bonus(eye_rgb, proprio_vec, helper: RegardHelper):
    m_adv = helper.motion_center_adv(eye_rgb)
    s = np.tanh(0.5 * estimate_wrist_speed_from_obs(proprio_vec))
    return max(0.0, m_adv) * s



# use the tanh function to limit the range of motion (conforming to physical constraints), restricting it to -1 to 1.
class TanhNormal:
    def __init__(self, mu, log_std):
        self.mu = mu
        self.log_std = log_std.clamp(-5, 2)#限制范围，避免梯度爆炸
        self.std = torch.exp(self.log_std)
        self.normal = torch.distributions.Normal(self.mu, self.std)
    def sample(self):
        x = self.normal.rsample()
        a = torch.tanh(x)
        log_prob = self.normal.log_prob(x) - torch.log(1 - a.pow(2) + 1e-6)
        return a, log_prob.sum(-1)
    def log_prob(self, a): #加个修正，让输出更平滑
        atanh = 0.5 * (torch.log1p(a + 1e-6) - torch.log1p(-a + 1e-6))
        log_prob = self.normal.log_prob(atanh) - torch.log(1 - a.pow(2) + 1e-6)
        return log_prob.sum(-1)
    @property
    def entropy(self):
        return (0.5 + 0.5 * torch.log(2 * torch.pi * self.std.pow(2))).sum(-1)

# combine
class PPOPolicy(nn.Module):
    def __init__(self, enc, pv, act_low, act_high, init_log_std=-0.2, device="cpu"):
        super().__init__()
        self.enc = enc
        self.pv  = pv
        self.log_std = nn.Parameter(torch.ones(act_low.shape[0]) * init_log_std)
        self.register_buffer("a_low",  torch.tensor(act_low, dtype=torch.float32))#aciton space limit
        self.register_buffer("a_high", torch.tensor(act_high, dtype=torch.float32))
        self.device = device
    def _obs_to_z(self, obs):
        left  = obs["eye_left"]#left eye and right eye
        right = obs["eye_right"]
        eye = np.concatenate([left, right], axis=2)
        eye_t = to_torch_contig(eye, self.device).permute(0,3,1,2)
        proprio  = to_torch_contig(obs["observation"], self.device)#observation
        touch    = to_torch_contig(obs["touch"], self.device)#touch
        return self.enc({"proprio":proprio, "tactile":touch, "rgb":eye_t})
    @torch.no_grad()#save space
    def act(self, obs):
        z = self._obs_to_z(obs)
        mu, v = self.pv(z)
        dist = TanhNormal(mu, self.log_std)#use tanh to -1,1
        a_norm, logp = dist.sample()
        center = (self.a_high + self.a_low)/2
        scale  = (self.a_high - self.a_low)/2
        a = center + scale * a_norm
        return a.squeeze(0).cpu().numpy().astype(np.float32), logp.item(), v.squeeze(0).item(), z
    @torch.no_grad()
    def act_deterministic(self, obs):#use tanh for evaluation
        z = self._obs_to_z(obs)
        mu, _ = self.pv(z)
        a_norm = torch.tanh(mu)
        center = (self.a_high + self.a_low)/2
        scale  = (self.a_high - self.a_low)/2
        a = center + scale * a_norm
        return a.squeeze(0).cpu().numpy().astype(np.float32)

#buffer (observation+action+reward+value+log action prob + wether it is done
class RolloutBuffer:
    def __init__(self, size, obs_keep=("eye_left","eye_right","observation","touch")):
        self.size = size#maximunm size
        self.data = {k: [] for k in ["obs","act","rew","val","logp","done","z"]}
        self.obs_keep = obs_keep
    def add(self, obs, act, rew, val, logp, done, z):
        self.data["obs"].append({k: obs[k] for k in self.obs_keep})
        self.data["act"].append(act)
        self.data["rew"].append(rew)
        self.data["val"].append(val)
        self.data["logp"].append(logp)
        self.data["done"].append(done)
        self.data["z"].append(z.detach().cpu())
    def stack(self):
        return {
            "obs": self.data["obs"],
            "act": np.asarray(self.data["act"], dtype=np.float32),
            "rew": np.asarray(self.data["rew"], dtype=np.float32),
            "val": np.asarray(self.data["val"], dtype=np.float32),
            "logp": np.asarray(self.data["logp"], dtype=np.float32),
            "done": np.asarray(self.data["done"], dtype=np.float32),
            "z": self.data["z"],
        }

#GAE
def compute_gae(rew, val, done, gamma=0.99, lam=0.95):
    T = len(rew)
    adv = np.zeros(T, np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):#caculated the advantage of each step
        next_nonterm = 1.0 - float(done[t])
        next_v = val[t+1] if t < T-1 else 0.0 # value in next step
        delta = rew[t] + gamma * next_v * next_nonterm - val[t] #TD difference=reward(now) + y x value - value(now)
        lastgaelam = delta + gamma * lam * next_nonterm * lastgaelam#TD sum
        adv[t] = lastgaelam
    ret = adv + val
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret


class EMANorm:#Reward Normalization (Preventing excessive or insufficient reward scales)
    def __init__(self, beta=0.01):
        self.mean = 0.0
        self.var = 1.0
        self.beta = beta
        self.inited = False
    def update(self, x):
        if not self.inited:
            self.mean = float(x)
            self.var = 1.0
            self.inited = True
        else:
            d = float(x) - self.mean
            self.mean += self.beta * d
            self.var = (1 - self.beta) * self.var + self.beta * d * d
    def norm(self, x):
        return (float(x) - self.mean) / (np.sqrt(self.var) + 1e-6)

#ppo
def ppo_update(enc, pv, ppo_policy, optimizer, buffer, epochs=10,
               batch_size=4096, clip_ratio=0.2, vf_coef=0.5, ent_coef=0.001):
    data = buffer.stack() #get data from buffer
    T = len(data["rew"])
    device = next(enc.parameters()).device

    z_all = torch.cat(data["z"]).to(device)#data from encoder
    act   = torch.tensor(data["act"], dtype=torch.float32, device=device)
    old_logp = torch.tensor(data["logp"], dtype=torch.float32, device=device)
    val  = torch.tensor(data["val"], dtype=torch.float32, device=device)
    rew  = data["rew"]
    done = data["done"]

    adv, ret = compute_gae(rew, val.cpu().numpy(), done)
    adv = torch.tensor(adv, dtype=torch.float32, device=device)
    ret = torch.tensor(ret, dtype=torch.float32, device=device)

    with torch.no_grad():
        mu_all, _ = pv(z_all)
    dist_all = TanhNormal(mu_all, ppo_policy.log_std)
    center = (ppo_policy.a_high + ppo_policy.a_low)/2
    scale  = (ppo_policy.a_high - ppo_policy.a_low)/2
    a_norm = torch.clamp((act - center)/(scale + 1e-6), -0.999, 0.999)

    idxs = np.arange(T)
    for _ in range(epochs):
        np.random.shuffle(idxs)
        for start in range(0, T, batch_size):
            mb_idx = idxs[start:start+batch_size]
            mb = torch.tensor(mb_idx, dtype=torch.long, device=device)

            mu, v = pv(z_all[mb])
            dist = TanhNormal(mu, ppo_policy.log_std)#current strategy
            logp = dist.log_prob(a_norm[mb]) #new strategy for previous  action
            ratio = torch.exp(logp - old_logp[mb])

            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv[mb]
            pi_loss = -torch.min(surr1, surr2).mean()
            v_loss = 0.5 * (ret[mb] - v).pow(2).mean()
            ent = dist.entropy.mean()

            loss = pi_loss + vf_coef * v_loss - ent_coef * ent#loss= strategy - critic

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(pv.parameters()) + [ppo_policy.log_std], 1.0)
            optimizer.step()

#RND
class RNDTrainer:
    def __init__(self, rnd, lr=5e-6, batch_size=256, epochs=3):
        self.rnd = rnd
        self.opt = torch.optim.Adam(rnd.predictor.parameters(), lr=lr)
        self.batch_size = batch_size
        self.epochs = epochs
        self.buf_z = []
        self.buf_t = []
    def push(self, z):
        with torch.no_grad():
            t = self.rnd.target(z).detach().cpu()
        self.buf_z.append(z.detach().cpu())
        self.buf_t.append(t)
    def train_if_ready(self, device):
        if len(self.buf_z) < self.batch_size:#if size insufficient, it will directly return
            return None
        z = torch.cat(self.buf_z, dim=0).to(device)
        t = torch.cat(self.buf_t, dim=0).to(device)
        idx = np.arange(z.size(0)) #Shuffle the index
        losses = []
        for _ in range(self.epochs):
            np.random.shuffle(idx) #Shuffle the order to ensure each epoch sees a different sequence of samples, thereby preventing training bias.
            for s in range(0, len(idx), self.batch_size):
                mb = idx[s:s+self.batch_size]
                mb = torch.tensor(mb, dtype=torch.long, device=device)
                p = self.rnd.predictor(z[mb])
                loss = torch.mean((p - t[mb])**2) #loss = predictor - target
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                losses.append(loss.item())
        self.buf_z.clear(); self.buf_t.clear()
        return float(np.mean(losses)) if losses else None

# to video
def get_rgb_frame(env, obs, both_eyes=True):
    try:
        env.render_mode = "rgb_array"
        fr = env.render()
        if isinstance(fr, np.ndarray) and fr.ndim == 3 and fr.shape[2] == 3:
            return np.ascontiguousarray(fr.astype(np.uint8))
    except Exception:
        pass
    try:
        renderer = getattr(env, "mujoco_renderer", None) or getattr(getattr(env, "unwrapped", env), "mujoco_renderer", None)
        if renderer is not None:
            fr = renderer.render(render_mode="rgb_array")
            if isinstance(fr, np.ndarray) and fr.ndim == 3 and fr.shape[2] == 3:
                return np.ascontiguousarray(fr.astype(np.uint8))
    except Exception:
        pass
    left = np.ascontiguousarray(obs["eye_left"]).astype(np.uint8)
    if both_eyes:
        right = np.ascontiguousarray(obs["eye_right"]).astype(np.uint8)
        rgb = np.hstack([left, right])
    else:
        rgb = left
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = cv2.convertScaleAbs(bgr, alpha=2.0, beta=-200)
    H, W = bgr.shape[:2]
    cv2.drawMarker(bgr, (W//2, H//2), (0,255,0), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def record_eval_video(policy_eval, steps=600, out_path="videos/babybench_eval.mp4", fps=30):
    env = gym.make("BabyBench")
    ensure_dir(out_path)
    obs, _ = env.reset()
    frame0 = get_rgb_frame(env, obs, both_eyes=True)
    H, W = frame0.shape[:2]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (W, H))
    if not vw.isOpened():
        out_path = os.path.splitext(out_path)[0] + ".avi"
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
        print("fallback MJPG:", out_path)
    vw.write(cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR))
    for t in range(1, steps+1):
        a = policy_eval.act_deterministic(obs)
        obs, _, term, trunc, _ = env.step(a)
        fr = get_rgb_frame(env, obs, both_eyes=True)
        if fr.shape[:2] != (H, W):
            fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_NEAREST)
        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, f"eval frame {t}", (6, H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        vw.write(bgr)
        if term or trunc:
            break
    vw.release(); env.close()
    print("saved video:", out_path)

# traning function
def main(
    total_env_steps=1_000_000,
    horizon=4096,
    ppo_epochs=10,
    batch_size=4096,
    gamma=0.99, lam=0.95,
    clip_ratio=0.2, vf_coef=0.5, ent_coef=0.005,
    rnd_lr=5e-6, rnd_batch_size=256, rnd_epochs=3,
    warmup_steps=800,
    phase1_until=300_000,
    # hand regard/bring hands into view
    w_cur_1=1.0, w_touch_1=0.2, w_reg_1=1.4, w_bi_1=0.3, w_sym_1=0.05,
    #  self-touch
    w_cur_2=1.0, w_touch_2=1.2, w_reg_2=0.6, w_bi_2=1.5, w_sym_2=0.10,
    save_every=50_000,
    eval_video_steps=800,
    ckpt_dir="checkpoints",#checkpoint
    log_path="logs/ppo_training.pkl",
    ppo_lr=3e-4
):
    os.makedirs("logs", exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("BabyBench")
    obs, _ = env.reset()
    proprio_dim = int(np.array(obs["observation"]).size)
    tactile_dim = int(np.array(obs["touch"]).size)
    H, W, C = np.array(obs["eye_left"]).shape
    visual_shape = (C * 2, H, W)
    act_dim = int(env.action_space.shape[0])

    enc = Encoder(proprio_dim, tactile_dim, visual_shape, out_dim=256).to(device)
    pv  = PolicyValue(in_dim=256, act_dim=act_dim).to(device)
    rnd = RND(in_dim=256).to(device)

    ppo_policy = PPOPolicy(enc, pv, env.action_space.low, env.action_space.high, init_log_std=-0.2, device=device)
    optim = torch.optim.Adam(list(enc.parameters()) + list(pv.parameters()) + [ppo_policy.log_std], lr=ppo_lr, weight_decay=1e-5)

    rnd_trainer = RNDTrainer(rnd, lr=rnd_lr, batch_size=rnd_batch_size, epochs=rnd_epochs)
    rnd_norm = EMANorm(beta=0.01)

    touch_stats = TouchStats(dim=tactile_dim, beta=0.01)
    regard = RegardHelper(H=H, W=W, crop=0.5)
    bimanual_det = BimanualTouchDetector(peak_thresh=2.0, window=10)

    logs = {"avg_rnd":[], "avg_touch":[], "avg_regard":[], "avg_bi":[], "avg_total":[], "step":[], "rnd_loss":[]}
    step = 0
    ep_rnd = ep_touch = ep_reg = ep_bi = ep_total = 0.0
    ep_steps = 0
    t0 = time.time()
    o, _ = env.reset()

    while step < total_env_steps:
        buffer = RolloutBuffer(horizon)
        rnd_losses_batch = []

        for t in range(horizon):
            # curriculum weights
            if step < phase1_until:#encourage bring hand in to visual field
                w_cur, w_touch, w_reg, w_bi, w_sym = w_cur_1, w_touch_1, w_reg_1, w_bi_1, w_sym_1
                cur_ent_coef = ent_coef
            else:#encourage self touch
                w_cur, w_touch, w_reg, w_bi, w_sym = w_cur_2, w_touch_2, w_reg_2, w_bi_2, w_sym_2
                cur_ent_coef = 0.001  # lower exploration in phase2

            a, logp, v, z = ppo_policy.act(o)
            o2, _, term, trunc, _ = env.step(a)
            done = term or trunc

            # RND reward \
            raw_rnd = rnd.intrinsic_reward(z).item()
            rnd_norm.update(raw_rnd)
            r_rnd = rnd_norm.norm(raw_rnd)
            r_rnd = float(np.clip(r_rnd, -3.0, 3.0))
            r_rnd = max(0.0, r_rnd)

            # Touch reward: compute before updating stats
            r_touch = touch_bonus(o2["touch"], touch_stats, dead=0.1, peak_thresh=2.0) if step > warmup_steps else 0.0
            touch_stats.update(o2["touch"])

            # Hand regard
            r_reg   = hand_regard_bonus(o2["eye_left"], o2["observation"], regard)

            # Total reward
            r_total = (w_cur*r_rnd + w_touch*r_touch + w_reg*r_reg)

            buffer.add(o, a, r_total, v, logp, done, z)

            # RND mini batch accumulation(for stablize
            rnd_trainer.push(z)
            rnd_loss_mean = rnd_trainer.train_if_ready(device)
            if rnd_loss_mean is not None:
                rnd_losses_batch.append(rnd_loss_mean)

            # episode stats
            ep_rnd += r_rnd; ep_touch += r_touch; ep_reg += r_reg; ep_bi += r_bi; ep_total += r_total; ep_steps += 1

            # debug touch diagnostics
            if step % 5000 == 0:
                x = o2["touch"].astype(np.float32)
                zt = (x - touch_stats.mu) / (touch_stats.sigma + 1e-6)
                rms_now = float(np.sqrt((np.clip(zt, -5.0, 5.0) ** 2).mean()))
                max_abs = float(np.max(np.abs(zt))) if zt.size > 0 else 0.0
                print(f"[dbg] step={step} touch_r={r_touch:.3f} bi={r_bi:.1f} rms={rms_now:.3f} max|z|={max_abs:.2f} RND={r_rnd:.3f}")

            o = o2; step += 1
            if done:
                avg_rnd = ep_rnd/ep_steps; avg_touch = ep_touch/ep_steps; avg_reg = ep_reg/ep_steps; avg_bi = ep_bi/ep_steps; avg_total = ep_total/ep_steps
                logs["avg_rnd"].append(avg_rnd); logs["avg_touch"].append(avg_touch); logs["avg_regard"].append(avg_reg); logs["avg_bi"].append(avg_bi); logs["avg_total"].append(avg_total); logs["step"].append(step)
                print(f"[ep end] step={step} avg: RND={avg_rnd:.4f} Touch={avg_touch:.4f} Regard={avg_reg:.4f} Bi={avg_bi:.4f} Total={avg_total:.4f}")
                ep_rnd = ep_touch = ep_reg = ep_bi = ep_total = 0.0; ep_steps = 0
                o, _ = env.reset()

            if step % 10000 == 0:
                dt = time.time()-t0
                if rnd_losses_batch:
                    print(f"[progress] step={step}/{total_env_steps} time={dt/60:.1f} min | RND_loss_mean={np.mean(rnd_losses_batch):.6f}")

            if step >= total_env_steps:
                break

        # PPO update (ENcofer +policy +valu)
        ppo_update(enc, pv, ppo_policy, optim, buffer,
                   epochs=ppo_epochs, batch_size=batch_size,
                   clip_ratio=clip_ratio, vf_coef=vf_coef, ent_coef=cur_ent_coef)
        print(f"[ppo] step={step} updated.")

        if rnd_losses_batch:#check if it is fitting target
            logs["rnd_loss"].append(float(np.mean(rnd_losses_batch)))
        else:
            logs["rnd_loss"].append(None)

        # periodic save and eval
        if step % save_every == 0 or step >= total_env_steps:
            ckpt_path = os.path.join(ckpt_dir, f"ppo_step_{step}.pt")
            torch.save({
                "enc": enc.state_dict(),
                "pv": pv.state_dict(),
                "rnd_predictor": rnd.predictor.state_dict(),
                "rnd_target": rnd.target.state_dict(),
                "log_std": ppo_policy.log_std.data.cpu().numpy(),
                "env_action_low": env.action_space.low,
                "env_action_high": env.action_space.high,
                "step": step
            }, ckpt_path)
            print("saved ckpt:", ckpt_path)

            with open(log_path, "wb") as f:
                pickle.dump(logs, f)
            print("saved log:", log_path)

            record_eval_video(ppo_policy, steps=eval_video_steps, out_path=f"videos/eval_step_{step}.mp4", fps=30)

    env.close()
    print("training done. total steps:", step)

if __name__ == "__main__":
    main(
        total_env_steps=300_000,
        horizon=4096,
        ppo_epochs=10,
        batch_size=2048,
        rnd_lr=1e-5,
        rnd_batch_size=128,
        rnd_epochs=2,
        save_every=50_000
    )



#
#
#
# #ignore this
# class BimanualTouchDetector:
#     def __init__(self, peak_thresh=2.0, window=10):
#         self.peak_thresh = float(peak_thresh)
#         self.window = int(window)
#         self.bufL, self.bufR = [], []
#     def step(self, zL_maxabs, zR_maxabs):
#         self.bufL.append(zL_maxabs >= self.peak_thresh)
#         self.bufR.append(zR_maxabs >= self.peak_thresh)
#         if len(self.bufL) > self.window: self.bufL.pop(0)
#         if len(self.bufR) > self.window: self.bufR.pop(0)
#         return 1.0 if (any(self.bufL) and any(self.bufR)) else 0.0
#
# def split_hands_touch(x, idx_left=None, idx_right=None):
#     x = np.asarray(x, np.float32)
#     n = x.size; mid = n // 2
#     if idx_left is None or idx_right is None:
#         return x[:mid], x[mid:]
#     return x[idx_left], x[idx_right]
#
# def bimanual_touch_reward(touch_vec, stats, det: BimanualTouchDetector, z_clip=5.0):
#     z = (touch_vec.astype(np.float32) - stats.mu) / (stats.sigma + 1e-6)
#     if z_clip is not None:
#         z = np.clip(z, -z_clip, z_clip)
#     L, R = split_hands_touch(z)
#     zL_max = float(np.max(np.abs(L))) if L.size else 0.0
#     zR_max = float(np.max(np.abs(R))) if R.size else 0.0
#     return det.step(zL_max, zR_max)  # 0/1 event
#
# def symmetry_bonus(action):
#     a = np.asarray(action, np.float32)
#     n = a.size; mid = n // 2
#     L, R = a[:mid], a[mid:mid+len(a[:mid])]
#     diff = np.linalg.norm(L - R) / (np.linalg.norm(L) + np.linalg.norm(R) + 1e-6)
#     return float(np.tanh(1.0 - diff))  # [0,1)
