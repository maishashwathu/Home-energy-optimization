import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from stable_baselines3 import DQN
import gymnasium as gym
from gymnasium import spaces
import time

# ── Palette ────────────────────────────────────────────────
PALETTE = {
    'primary':  '#58a6ff',
    'success':  '#3fb950',
    'danger':   '#f85149',
    'warning':  '#d29922',
    'purple':   '#bc8cff',
}

APPLIANCE_NAMES = ['AC', 'Wash. Machine', 'Dishwasher', 'Fridge', 'Fan', 'Lights']

APPLIANCE_POWER = {
    'ac':     2.0,
    'wm':     1.5,
    'dw':     1.2,
    'fridge': 0.2,
    'fan':    0.1,
    'lights': 0.3,
}

BASE_PRICE = np.array([0.06, 0.05, 0.05, 0.05, 0.06, 0.07,
                       0.09, 0.12, 0.15, 0.14, 0.13, 0.14,
                       0.16, 0.18, 0.20, 0.19, 0.17, 0.16,
                       0.14, 0.13, 0.11, 0.10, 0.08, 0.07])

PRICE_PROFILE = BASE_PRICE


# ── Environment (kept in sync with notebook) ───────────────
class SmartHomeEnv(gym.Env):
    """
    Smart Home Energy Management Environment.

    Observations:
        [hour_norm, price_norm, ac, wm, dw, fridge, fan, lights]  → shape (8,)

    Actions:
        0-7  → binary encoding of (AC, WM, DW)

    Reward:
        -cost  - comfort_penalty  - peak_penalty  - switch_penalty
        + off_peak_bonus  - end_of_day_penalty
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, price_profile=None, render_mode=None):
        super().__init__()

        self.price_profile = price_profile if price_profile is not None else PRICE_PROFILE
        self.render_mode   = render_mode

        # ── Spaces ─────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(8)

        # ── Internal state ──────────────────────────────────
        self.hour         = 0
        self.prev_action  = np.array([0, 0, 0], dtype=np.int32)
        self.episode_cost = 0.0
        self.episode_len  = 24

        # ── Task completion tracking ────────────────────────
        self.wm_ran_today = False
        self.dw_ran_today = False

        # ── Logging ─────────────────────────────────────────
        self.history = []

    def _decode_action(self, action: int):
        ac = (action >> 2) & 1
        wm = (action >> 1) & 1
        dw = (action >> 0) & 1
        return np.array([ac, wm, dw], dtype=np.int32)

    def _fixed_appliances(self, hour: int):
        fridge = 1
        fan    = 1 if 9 <= hour < 21 else 0
        lights = 1 if (hour >= 18 or hour < 6) else 0
        return np.array([fridge, fan, lights], dtype=np.float32)

    def _build_obs(self, hour, controlled):
        hour_norm  = hour / 23.0
        price_norm = (self.price_profile[hour] - self.price_profile.min()) / \
                     (self.price_profile.max() - self.price_profile.min() + 1e-8)
        fixed = self._fixed_appliances(hour)
        obs   = np.concatenate([
            [hour_norm, price_norm],
            controlled.astype(np.float32),
            fixed
        ]).astype(np.float32)
        return obs

    def _compute_reward(self, hour, controlled, prev_controlled):
        price = self.price_profile[hour]
        fixed = self._fixed_appliances(hour)

        # ── Energy cost ──────────────────────────────────────
        all_states = np.concatenate([controlled, fixed])
        powers     = np.array([
            APPLIANCE_POWER['ac'],
            APPLIANCE_POWER['wm'],
            APPLIANCE_POWER['dw'],
            APPLIANCE_POWER['fridge'],
            APPLIANCE_POWER['fan'],
            APPLIANCE_POWER['lights'],
        ])
        total_power = float(np.dot(all_states, powers))
        cost        = total_power * price

        # ── Comfort penalty ──────────────────────────────────
        comfort_penalty = 0.0
        if 12 <= hour < 16 and controlled[0] == 0:
            comfort_penalty = 0.5

        # ── Peak usage penalty ───────────────────────────────
        peak_penalty = 0.0
        if price > 0.15 and controlled[0] == 1 and controlled[1] == 1:
            peak_penalty = 0.3

        # ── Switching penalty ────────────────────────────────
        switches       = np.sum(np.abs(controlled - prev_controlled))
        switch_penalty = 0.05 * switches

        # ── Off-peak bonus for WM and DW ─────────────────────
        off_peak_bonus = 0.0
        if price <= 0.09:
            if controlled[1] == 1:
                off_peak_bonus += 0.25
            if controlled[2] == 1:
                off_peak_bonus += 0.25

        # ── End-of-day penalty if WM/DW never ran ────────────
        end_of_day_penalty = 0.0
        if hour == 23:
            wm_ran = self.wm_ran_today or (controlled[1] == 1)
            dw_ran = self.dw_ran_today or (controlled[2] == 1)
            if not wm_ran:
                end_of_day_penalty += 1.0
            if not dw_ran:
                end_of_day_penalty += 1.0

        reward = (-cost
                  - comfort_penalty
                  - peak_penalty
                  - switch_penalty
                  + off_peak_bonus
                  - end_of_day_penalty)

        info = {
            'cost':               cost,
            'comfort_penalty':    comfort_penalty,
            'peak_penalty':       peak_penalty,
            'switch_penalty':     switch_penalty,
            'off_peak_bonus':     off_peak_bonus,
            'end_of_day_penalty': end_of_day_penalty,
            'total_power':        total_power,
            'price':              price,
            'controlled':         controlled.copy(),
            'fixed':              fixed.copy(),
            'hour':               hour,
        }
        return reward, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.hour         = 0
        self.prev_action  = np.array([0, 0, 0], dtype=np.int32)
        self.episode_cost = 0.0
        self.history      = []
        self.wm_ran_today = False
        self.dw_ran_today = False

        obs = self._build_obs(self.hour, self.prev_action)
        return obs, {}

    def step(self, action: int):
        controlled   = self._decode_action(int(action))
        reward, info = self._compute_reward(self.hour, controlled, self.prev_action)

        if controlled[1] == 1:
            self.wm_ran_today = True
        if controlled[2] == 1:
            self.dw_ran_today = True

        self.episode_cost += info['cost']
        self.history.append(info)
        self.prev_action   = controlled.copy()
        self.hour         += 1

        terminated = (self.hour >= self.episode_len)
        truncated  = False

        if not terminated:
            obs = self._build_obs(self.hour, controlled)
        else:
            obs = self._build_obs(0, np.zeros(3, dtype=np.int32))

        return obs, reward, terminated, truncated, info


# ── Streamlit UI ────────────────────────────────────────────
st.set_page_config(page_title='Smart Home RL Dashboard', page_icon='⚡', layout='wide')
st.title('🏠⚡ Smart Home Energy Management – RL Dashboard')
st.markdown('---')

# Sidebar controls
st.sidebar.header('⚙️ Settings')
price_mult = st.sidebar.slider('Price Multiplier', 0.5, 3.0, 1.0, 0.1)
show_live  = st.sidebar.checkbox('Live Simulation Mode', value=True)

# ── Load or train model ─────────────────────────────────────
# ⚠️ Bump version number every time you change the environment or reward
@st.cache_resource
def get_model(version=4):
    env = SmartHomeEnv(BASE_PRICE)
    m   = DQN('MlpPolicy', env, learning_rate=1e-3, buffer_size=50000,
              verbose=0, seed=42)
    m.learn(100000)
    return m

with st.spinner('Training DQN agent … (first load only)'):
    model = get_model(version=4)

price = BASE_PRICE * price_mult


# ── Run RL & Baseline ───────────────────────────────────────
def run(policy, price_profile):
    env    = SmartHomeEnv(price_profile)
    obs, _ = env.reset()
    done   = False
    while not done:
        if policy == 'rl':
            a, _ = model.predict(obs, deterministic=True)
        else:
            a = 7
        obs, _, done, _, _ = env.step(int(a))
    return env.history

rl_h  = run('rl',   price)
bl_h  = run('base', price)
rl_c  = [h['cost'] for h in rl_h]
bl_c  = [h['cost'] for h in bl_h]
hours = list(range(24))

bl_total    = sum(bl_c)
rl_total    = sum(rl_c)
savings_pct = ((bl_total - rl_total) / bl_total * 100) if bl_total > 0 else 0.0


# ── Metrics row ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric('Baseline Cost',      f"${bl_total:.4f}")
c2.metric('RL Agent Cost',      f"${rl_total:.4f}",
          delta=f"-${bl_total - rl_total:.4f}", delta_color='normal')
c3.metric('Savings',            f"{savings_pct:.1f}%")
c4.metric('Comfort Violations', sum(1 for h in rl_h if h['comfort_penalty'] > 0))

st.markdown('---')

# ── Charts ──────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader('📈 Energy Cost Comparison')
    df = pd.DataFrame({'Hour': hours, 'Baseline': bl_c, 'RL Agent': rl_c})
    st.line_chart(df.set_index('Hour'), color=[PALETTE['danger'], PALETTE['success']])

with col2:
    st.subheader('⚡ Price Profile')
    df_p = pd.DataFrame({'Hour': hours, 'Price': price})
    st.line_chart(df_p.set_index('Hour'), color=[PALETTE['purple']])

# ── Appliance heatmap ────────────────────────────────────────
st.subheader('🔥 Appliance Usage Heatmap')
mat = np.zeros((6, 24))
for t, h in enumerate(rl_h):
    mat[:, t] = np.concatenate([h['controlled'], h['fixed']])
fig, ax = plt.subplots(figsize=(16, 4))
sns.heatmap(mat, ax=ax, cmap='YlGn',
            xticklabels=[f'{i:02d}' for i in hours],
            yticklabels=APPLIANCE_NAMES,
            linewidths=0.5, vmin=0, vmax=1, annot=True, fmt='.0f')
st.pyplot(fig)
plt.close(fig)

# ── Live simulation ──────────────────────────────────────────
if show_live:
    st.subheader('▶️ Live Simulation Playback')
    if st.button('▶ Play 24-Hour Replay'):
        placeholder = st.empty()
        for t in range(24):
            h      = rl_h[t]
            ctrl   = h['controlled']
            fixed  = h['fixed']
            states = dict(zip(APPLIANCE_NAMES, np.concatenate([ctrl, fixed]).astype(int)))
            with placeholder.container():
                st.markdown(
                    f"### Hour {t:02d}:00  |  "
                    f"Price: ${h['price']:.3f}/kWh  |  "
                    f"Cost: ${h['cost']:.4f}"
                )
                cols = st.columns(6)
                for col, (name, state) in zip(cols, states.items()):
                    col.metric(name, '🟢 ON' if state else '🔴 OFF')
            time.sleep(0.4)
        placeholder.success('✅ Replay complete!')