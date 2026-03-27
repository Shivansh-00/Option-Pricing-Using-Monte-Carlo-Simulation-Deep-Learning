"""
Reinforcement Learning for Dynamic Hedging
============================================
PPO and DQN agents for optimal delta-hedging of option portfolios.

State space:  [S/K, σ_impl, Δ, Γ, Θ, regime_id, current_hedge, P&L]
Action space: Discrete hedge adjustments or continuous hedge ratio
Reward:       -|P&L variance| - λ_tc * transaction_costs + λ_sr * risk_adj_return

Integrates with:
    - PINNs / pricing engine for live Greeks
    - Regime detector for regime-aware hedging
    - Portfolio risk dashboard for aggregate exposure
"""

from __future__ import annotations
import numpy as np
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from scipy.stats import norm

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
#  GBM + Jump Diffusion Environment
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class HedgingEnvConfig:
    S0: float = 100.0
    K: float = 100.0
    r: float = 0.05
    sigma: float = 0.20
    T: float = 1.0           # option maturity in years
    dt: float = 1 / 252      # daily rebalancing
    transaction_cost: float = 0.001  # 10 bps per trade
    jump_intensity: float = 0.0      # Merton jump diffusion
    jump_mean: float = -0.05
    jump_std: float = 0.10
    regime_enabled: bool = True


class HedgingEnvironment:
    """
    Simulates the environment for a delta-hedging agent.
    The agent holds a short call option and hedges with the underlying.
    """

    def __init__(self, config: Optional[HedgingEnvConfig] = None, seed: int = 42):
        self.cfg = config or HedgingEnvConfig()
        self.rng = np.random.default_rng(seed)
        self.n_steps = int(self.cfg.T / self.cfg.dt)
        self.reset()

    def reset(self) -> np.ndarray:
        self.step_idx = 0
        self.S = self.cfg.S0
        self.hedge_ratio = 0.0
        self.cash = 0.0
        self.pnl_history: List[float] = []
        self.price_path: List[float] = [self.S]

        # Regime state (0=bull, 1=bear, 2=crisis)
        self.regime = 0
        self._regime_transition = np.array([
            [0.95, 0.04, 0.01],
            [0.05, 0.90, 0.05],
            [0.02, 0.08, 0.90]
        ])
        self._regime_vols = [self.cfg.sigma, self.cfg.sigma * 1.5, self.cfg.sigma * 2.5]

        return self._get_state()

    def _bs_delta(self, S: float, K: float, tau: float, sigma: float, r: float) -> float:
        if tau <= 0:
            return 1.0 if S > K else 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
        return float(norm.cdf(d1))

    def _bs_price(self, S: float, K: float, tau: float, sigma: float, r: float) -> float:
        if tau <= 0:
            return max(S - K, 0)
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
        d2 = d1 - sigma * math.sqrt(tau)
        return float(S * norm.cdf(d1) - K * math.exp(-r * tau) * norm.cdf(d2))

    def _bs_gamma(self, S: float, K: float, tau: float, sigma: float, r: float) -> float:
        if tau <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
        return float(norm.pdf(d1) / (S * sigma * math.sqrt(tau)))

    def _bs_theta(self, S: float, K: float, tau: float, sigma: float, r: float) -> float:
        """Correct Black-Scholes theta for a call option."""
        if tau <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
        d2 = d1 - sigma * math.sqrt(tau)
        term1 = -(S * sigma * norm.pdf(d1)) / (2 * math.sqrt(tau))
        term2 = -r * K * math.exp(-r * tau) * norm.cdf(d2)
        return float(term1 + term2)

    def _get_state(self) -> np.ndarray:
        tau = max((self.n_steps - self.step_idx) * self.cfg.dt, 1e-6)
        sigma = self._regime_vols[self.regime] if self.cfg.regime_enabled else self.cfg.sigma

        delta = self._bs_delta(self.S, self.cfg.K, tau, sigma, self.cfg.r)
        gamma = self._bs_gamma(self.S, self.cfg.K, tau, sigma, self.cfg.r)
        theta = self._bs_theta(self.S, self.cfg.K, tau, sigma, self.cfg.r)

        moneyness = self.S / self.cfg.K
        pnl_so_far = sum(self.pnl_history) if self.pnl_history else 0.0

        return np.array([
            moneyness, sigma, delta, gamma, theta / 100,
            self.regime / 2.0, self.hedge_ratio, pnl_so_far / self.cfg.S0
        ], dtype=np.float64)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        action: 0-10 → hedge ratio from 0.0 to 1.0 in increments of 0.1
        """
        new_hedge = action / 10.0
        tau = max((self.n_steps - self.step_idx) * self.cfg.dt, 1e-6)
        sigma = self._regime_vols[self.regime] if self.cfg.regime_enabled else self.cfg.sigma

        # Transaction cost
        hedge_change = abs(new_hedge - self.hedge_ratio)
        tc = hedge_change * self.S * self.cfg.transaction_cost

        # Evolve price
        old_S = self.S
        dW = self.rng.normal(0, math.sqrt(self.cfg.dt))
        drift = (self.cfg.r - 0.5 * sigma**2) * self.cfg.dt
        diffusion = sigma * dW

        # Jump component
        jump = 0.0
        if self.cfg.jump_intensity > 0:
            n_jumps = self.rng.poisson(self.cfg.jump_intensity * self.cfg.dt)
            if n_jumps > 0:
                jump = sum(self.rng.normal(self.cfg.jump_mean, self.cfg.jump_std) for _ in range(n_jumps))

        self.S = old_S * math.exp(drift + diffusion + jump)
        self.S = max(self.S, 0.01)
        self.price_path.append(self.S)

        # Regime transition
        if self.cfg.regime_enabled:
            self.regime = self.rng.choice(3, p=self._regime_transition[self.regime])

        # P&L: option value change + hedge P&L - transaction costs
        old_V = self._bs_price(old_S, self.cfg.K, tau, sigma, self.cfg.r)
        new_tau = max(tau - self.cfg.dt, 0)
        new_V = self._bs_price(self.S, self.cfg.K, new_tau, sigma, self.cfg.r)
        option_pnl = -(new_V - old_V)  # short option
        hedge_pnl = new_hedge * (self.S - old_S)
        step_pnl = option_pnl + hedge_pnl - tc

        self.pnl_history.append(step_pnl)
        self.hedge_ratio = new_hedge
        self.step_idx += 1

        done = self.step_idx >= self.n_steps

        # Reward: penalise squared hedging error + transaction costs
        # Scale ×10 so SPSA gradient signal is not vanishingly small
        reward = (-step_pnl ** 2 - 2.0 * tc / self.cfg.S0) * 10.0

        info = {
            "step_pnl": step_pnl,
            "transaction_cost": tc,
            "spot": self.S,
            "regime": int(self.regime),
            "hedge_ratio": new_hedge,
            "option_pnl": option_pnl,
            "hedge_pnl": hedge_pnl
        }

        return self._get_state(), float(reward), done, info


# ═══════════════════════════════════════════════════════════════════════
#  Deep Q-Network Agent (pure NumPy)
# ═══════════════════════════════════════════════════════════════════════

class DQNAgent:
    """
    Deep Q-Network for discrete hedging actions.
    Actions: 11 levels (0.0, 0.1, ..., 1.0 hedge ratio)
    """

    def __init__(self, state_dim: int = 8, n_actions: int = 11,
                 hidden: int = 64, lr: float = 5e-3, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, seed: int = 42):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = np.random.default_rng(seed)

        # Q-network: 2 hidden layers
        scale1 = math.sqrt(2.0 / (state_dim + hidden))
        scale2 = math.sqrt(2.0 / (hidden + hidden))
        scale3 = math.sqrt(2.0 / (hidden + n_actions))

        self.W1 = self.rng.normal(0, scale1, (state_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = self.rng.normal(0, scale2, (hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.W3 = self.rng.normal(0, scale3, (hidden, n_actions))
        self.b3 = np.zeros(n_actions)

        # Target network (copy)
        self.W1_t = self.W1.copy()
        self.b1_t = self.b1.copy()
        self.W2_t = self.W2.copy()
        self.b2_t = self.b2.copy()
        self.W3_t = self.W3.copy()
        self.b3_t = self.b3.copy()

        # Replay buffer
        self.buffer: List[Tuple] = []
        self.buffer_max = 10000
        self.batch_size = 64

        self.train_steps = 0

    def _forward(self, state: np.ndarray, target: bool = False) -> np.ndarray:
        W1, b1 = (self.W1_t, self.b1_t) if target else (self.W1, self.b1)
        W2, b2 = (self.W2_t, self.b2_t) if target else (self.W2, self.b2)
        W3, b3 = (self.W3_t, self.b3_t) if target else (self.W3, self.b3)

        h1 = np.maximum(0, state @ W1 + b1)  # ReLU
        h2 = np.maximum(0, h1 @ W2 + b2)
        q_values = h2 @ W3 + b3
        return q_values

    def select_action(self, state: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        q = self._forward(state.reshape(1, -1))
        return int(np.argmax(q[0]))

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state.copy(), action, reward, next_state.copy(), done))
        if len(self.buffer) > self.buffer_max:
            self.buffer.pop(0)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        # Only train every 4 calls to reduce noise and speed up
        self.train_steps += 1
        if self.train_steps % 4 != 0:
            return

        idx = self.rng.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch], dtype=np.float64)

        # Q-targets
        q_next = self._forward(next_states, target=True)
        targets = rewards + self.gamma * np.max(q_next, axis=1) * (1 - dones)

        # Current Q
        q_current = self._forward(states)
        q_pred = q_current[np.arange(self.batch_size), actions]

        # TD error
        td_error = targets - q_pred

        # SPSA-style gradient update for Q-network
        pert = 5e-3
        all_grads = []
        for W, b in [(self.W1, self.b1), (self.W2, self.b2), (self.W3, self.b3)]:
            dW = self.rng.choice([-1.0, 1.0], size=W.shape)
            db = self.rng.choice([-1.0, 1.0], size=b.shape)

            W += pert * dW
            b += pert * db
            q_plus = self._forward(states)
            loss_plus = np.mean((targets - q_plus[np.arange(self.batch_size), actions]) ** 2)

            W -= 2 * pert * dW
            b -= 2 * pert * db
            q_minus = self._forward(states)
            loss_minus = np.mean((targets - q_minus[np.arange(self.batch_size), actions]) ** 2)

            W += pert * dW
            b += pert * db

            grad_W = (loss_plus - loss_minus) / (2 * pert * dW)
            grad_b = (loss_plus - loss_minus) / (2 * pert * db)

            all_grads.append((W, b, grad_W, grad_b))

        for W, b, grad_W, grad_b in all_grads:
            W -= self.lr * np.clip(grad_W, -1, 1)
            b -= self.lr * np.clip(grad_b, -1, 1)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Target network update (every 20 effective updates)
        if self.train_steps % 20 == 0:
            tau_soft = 0.1
            for Wm, Wt in [(self.W1, self.W1_t), (self.W2, self.W2_t), (self.W3, self.W3_t)]:
                Wt[:] = tau_soft * Wm + (1 - tau_soft) * Wt
            for bm, bt in [(self.b1, self.b1_t), (self.b2, self.b2_t), (self.b3, self.b3_t)]:
                bt[:] = tau_soft * bm + (1 - tau_soft) * bt


# ═══════════════════════════════════════════════════════════════════════
#  PPO Agent (pure NumPy)
# ═══════════════════════════════════════════════════════════════════════

class PPOAgent:
    """
    Proximal Policy Optimization for continuous hedge ratio.
    Actor outputs mean & log_std → Gaussian policy.
    Critic estimates state value V(s).
    """

    def __init__(self, state_dim: int = 8, hidden: int = 64, lr: float = 1e-3,
                 gamma: float = 0.99, gae_lambda: float = 0.95, clip_eps: float = 0.2,
                 entropy_coeff: float = 0.01, seed: int = 42):
        self.state_dim = state_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.lr = lr
        self.rng = np.random.default_rng(seed)

        s = math.sqrt(2.0 / (state_dim + hidden))
        # Actor
        self.actor_W1 = self.rng.normal(0, s, (state_dim, hidden))
        self.actor_b1 = np.zeros(hidden)
        self.actor_W2 = self.rng.normal(0, math.sqrt(2.0 / (hidden + 2)), (hidden, 2))  # mean, log_std
        self.actor_b2 = np.zeros(2)

        # Critic
        self.critic_W1 = self.rng.normal(0, s, (state_dim, hidden))
        self.critic_b1 = np.zeros(hidden)
        self.critic_W2 = self.rng.normal(0, math.sqrt(2.0 / (hidden + 1)), (hidden, 1))
        self.critic_b2 = np.zeros(1)

        self.trajectory: List[Dict] = []

    def _actor_forward(self, state: np.ndarray):
        h = np.tanh(state @ self.actor_W1 + self.actor_b1)
        out = h @ self.actor_W2 + self.actor_b2
        mean = 1.0 / (1.0 + np.exp(-out[..., 0]))  # sigmoid → [0, 1]
        log_std = np.clip(out[..., 1], -2, 0)
        return mean, log_std

    def _critic_forward(self, state: np.ndarray):
        h = np.tanh(state @ self.critic_W1 + self.critic_b1)
        return (h @ self.critic_W2 + self.critic_b2).ravel()

    def select_action(self, state: np.ndarray) -> Tuple[float, float]:
        mean, log_std = self._actor_forward(state.reshape(1, -1))
        std = np.exp(log_std[0])
        action = float(np.clip(mean[0] + self.rng.normal(0, max(std, 0.01)), 0, 1))
        log_prob = -0.5 * ((action - mean[0]) / (std + 1e-8))**2 - log_std[0] - 0.5 * math.log(2 * math.pi)
        return action, float(log_prob)

    def store(self, state, action, reward, log_prob, value, done):
        self.trajectory.append({
            "state": state.copy(), "action": action, "reward": reward,
            "log_prob": log_prob, "value": value, "done": done
        })

    def get_value(self, state: np.ndarray) -> float:
        return float(self._critic_forward(state.reshape(1, -1))[0])

    def update(self):
        if len(self.trajectory) < 10:
            return

        states = np.array([t["state"] for t in self.trajectory])
        actions = np.array([t["action"] for t in self.trajectory])
        rewards = np.array([t["reward"] for t in self.trajectory])
        old_log_probs = np.array([t["log_prob"] for t in self.trajectory])
        values = np.array([t["value"] for t in self.trajectory])
        dones = np.array([t["done"] for t in self.trajectory], dtype=np.float64)

        # GAE
        n = len(rewards)
        advantages = np.zeros(n)
        last_gae = 0
        for t in reversed(range(n)):
            next_val = values[t + 1] if t + 1 < n else 0
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + values

        # Normalise advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # SPSA update for actor and critic
        pert = 5e-3
        all_params = [
            (self.actor_W1, self.actor_b1), (self.actor_W2, self.actor_b2),
            (self.critic_W1, self.critic_b1), (self.critic_W2, self.critic_b2)
        ]

        def _total_loss():
            mean, log_std = self._actor_forward(states)
            std = np.exp(log_std)
            new_log_probs = -0.5 * ((actions - mean) / (std + 1e-8))**2 - log_std - 0.5 * math.log(2 * math.pi)
            ratio = np.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -np.mean(np.minimum(surr1, surr2))
            v_pred = self._critic_forward(states)
            critic_loss = np.mean((returns - v_pred) ** 2)
            return actor_loss + 0.5 * critic_loss

        all_grads = []
        for W, b in all_params:
            dW = self.rng.choice([-1.0, 1.0], size=W.shape)
            db = self.rng.choice([-1.0, 1.0], size=b.shape)

            W += pert * dW
            b += pert * db
            l_plus = _total_loss()

            W -= 2 * pert * dW
            b -= 2 * pert * db
            l_minus = _total_loss()

            W += pert * dW
            b += pert * db

            grad_W = (l_plus - l_minus) / (2 * pert * dW)
            grad_b = (l_plus - l_minus) / (2 * pert * db)

            all_grads.append((W, b, grad_W, grad_b))

        for W, b, grad_W, grad_b in all_grads:
            W -= self.lr * np.clip(grad_W, -1, 1)
            b -= self.lr * np.clip(grad_b, -1, 1)

        self.trajectory = []


# ═══════════════════════════════════════════════════════════════════════
#  Hedging Orchestrator
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class HedgingResult:
    total_pnl: float
    pnl_std: float
    sharpe: float
    max_drawdown: float
    avg_transaction_cost: float
    n_trades: int
    final_hedge: float
    regime_distribution: Dict[int, int]
    pnl_history: List[float]
    price_path: List[float]
    hedge_history: List[float]


class DynamicHedgingEngine:
    """Orchestrates RL-based dynamic hedging with backtesting."""

    def __init__(self, agent_type: str = "dqn", seed: int = 42):
        self.agent_type = agent_type
        self.seed = seed
        if agent_type == "dqn":
            self.agent = DQNAgent(seed=seed)
        else:
            self.agent = PPOAgent(seed=seed)
        self._trained = False
        self._train_history: List[Dict] = []

    def train(self, n_episodes: int = 200, env_config: Optional[HedgingEnvConfig] = None,
              progress_callback: Optional[Any] = None) -> Dict[str, Any]:
        """Train the RL agent. Optional progress_callback(episode, n_episodes, avg_reward)."""
        cfg = env_config or HedgingEnvConfig()
        self._train_history = []
        t0 = time.time()
        episode_rewards = []
        log_interval = max(1, n_episodes // 20)  # ~20 data points

        for ep in range(n_episodes):
            env = HedgingEnvironment(cfg, seed=self.seed + ep)
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                if self.agent_type == "dqn":
                    action = self.agent.select_action(state)
                    next_state, reward, done, info = env.step(action)
                    self.agent.store(state, action, reward, next_state, done)
                    self.agent.train_step()
                else:
                    action, log_prob = self.agent.select_action(state)
                    action_int = int(round(action * 10))
                    next_state, reward, done, info = env.step(action_int)
                    value = self.agent.get_value(state)
                    self.agent.store(state, action, reward, log_prob, value, done)

                state = next_state
                total_reward += reward

            if self.agent_type == "ppo":
                self.agent.update()

            episode_rewards.append(total_reward)

            if ep % log_interval == 0 or ep == n_episodes - 1:
                avg = float(np.mean(episode_rewards[-max(log_interval, 10):]))
                logger.info(f"Hedging RL episode {ep}/{n_episodes}: avg_reward={avg:.4f}")
                self._train_history.append({"episode": ep, "avg_reward": avg})
                if progress_callback:
                    try:
                        progress_callback(ep, n_episodes, avg)
                    except Exception:
                        pass

        self._trained = True
        elapsed = time.time() - t0
        return {
            "agent_type": self.agent_type,
            "episodes": n_episodes,
            "final_avg_reward": float(np.mean(episode_rewards[-max(log_interval, 10):])),
            "training_time_s": round(elapsed, 2),
            "history": self._train_history
        }

    def backtest(self, n_episodes: int = 100,
                 env_config: Optional[HedgingEnvConfig] = None) -> Dict[str, Any]:
        """Run backtest episodes and aggregate results."""
        cfg = env_config or HedgingEnvConfig()
        results: List[HedgingResult] = []

        # Also run BS-delta benchmark
        bs_pnls = []

        # Save epsilon and use pure exploitation during backtest
        saved_epsilon = None
        if self.agent_type == "dqn":
            saved_epsilon = self.agent.epsilon
            self.agent.epsilon = 0.0

        for ep in range(n_episodes):
            env = HedgingEnvironment(cfg, seed=self.seed + 10000 + ep)
            state = env.reset()
            done = False
            hedge_history = []
            regime_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0}

            while not done:
                if self.agent_type == "dqn":
                    action = self.agent.select_action(state)
                else:
                    action_cont, _ = self.agent.select_action(state)
                    action = int(round(action_cont * 10))
                next_state, reward, done, info = env.step(action)
                hedge_history.append(info["hedge_ratio"])
                regime_counts[info["regime"]] = regime_counts.get(info["regime"], 0) + 1
                state = next_state

            pnl_arr = np.array(env.pnl_history)
            total_pnl = float(pnl_arr.sum())
            pnl_std = float(pnl_arr.std()) if len(pnl_arr) > 1 else 0.0
            sharpe = total_pnl / (pnl_std + 1e-8) * math.sqrt(252)

            cum = np.cumsum(pnl_arr)
            dd = cum - np.maximum.accumulate(cum)
            max_dd = float(dd.min()) if len(dd) > 0 else 0.0

            results.append(HedgingResult(
                total_pnl=total_pnl, pnl_std=pnl_std, sharpe=sharpe,
                max_drawdown=max_dd, avg_transaction_cost=0.0,
                n_trades=len(hedge_history), final_hedge=hedge_history[-1] if hedge_history else 0,
                regime_distribution=regime_counts,
                pnl_history=env.pnl_history, price_path=env.price_path,
                hedge_history=hedge_history
            ))

            # BS delta benchmark
            env_bs = HedgingEnvironment(cfg, seed=self.seed + 10000 + ep)
            state_bs = env_bs.reset()
            done_bs = False
            while not done_bs:
                bs_delta = env_bs._bs_delta(env_bs.S, env_bs.cfg.K,
                                            max((env_bs.n_steps - env_bs.step_idx) * env_bs.cfg.dt, 1e-6),
                                            env_bs.cfg.sigma, env_bs.cfg.r)
                bs_action = int(round(bs_delta * 10))
                bs_action = min(max(bs_action, 0), 10)
                _, _, done_bs, _ = env_bs.step(bs_action)
            bs_pnls.append(sum(env_bs.pnl_history))

        # Restore epsilon after backtest
        if saved_epsilon is not None:
            self.agent.epsilon = saved_epsilon

        # Aggregate
        avg_pnl = float(np.mean([r.total_pnl for r in results]))
        avg_std = float(np.mean([r.pnl_std for r in results]))
        avg_sharpe = float(np.mean([r.sharpe for r in results]))
        avg_dd = float(np.mean([r.max_drawdown for r in results]))
        bs_avg_pnl = float(np.mean(bs_pnls))

        return {
            "agent_type": self.agent_type,
            "n_episodes": n_episodes,
            "rl_avg_pnl": round(avg_pnl, 4),
            "rl_avg_std": round(avg_std, 4),
            "rl_avg_sharpe": round(avg_sharpe, 4),
            "rl_max_drawdown": round(avg_dd, 4),
            "bs_delta_avg_pnl": round(bs_avg_pnl, 4),
            "improvement_vs_bs": round((avg_pnl - bs_avg_pnl) / (abs(bs_avg_pnl) + 1e-8) * 100, 2),
            "sample_path": {
                "price_path": results[0].price_path[:50],
                "pnl_history": results[0].pnl_history[:50],
                "hedge_history": results[0].hedge_history[:50]
            }
        }

    def suggest_hedge(self, state: Dict[str, float]) -> Dict[str, Any]:
        """Get hedge recommendation from trained agent."""
        s = np.array([
            state.get("moneyness", 1.0),
            state.get("implied_vol", 0.2),
            state.get("delta", 0.5),
            state.get("gamma", 0.02),
            state.get("theta", -0.01),
            state.get("regime", 0) / 2.0,
            state.get("current_hedge", 0.0),
            state.get("pnl", 0.0) / 100.0
        ])

        if self.agent_type == "dqn":
            q_values = self.agent._forward(s.reshape(1, -1))[0]
            action = int(np.argmax(q_values))
            confidence = float(np.exp(q_values[action]) / np.sum(np.exp(q_values)))
            return {
                "recommended_hedge_ratio": action / 10.0,
                "confidence": round(confidence, 4),
                "q_values": {f"hedge_{i/10:.1f}": round(float(q_values[i]), 4) for i in range(len(q_values))},
                "agent_type": "DQN"
            }
        else:
            mean, log_std = self.agent._actor_forward(s.reshape(1, -1))
            return {
                "recommended_hedge_ratio": round(float(mean[0]), 4),
                "uncertainty": round(float(np.exp(log_std[0])), 4),
                "agent_type": "PPO"
            }

    def get_status(self) -> Dict[str, Any]:
        return {
            "agent_type": self.agent_type,
            "trained": self._trained,
            "training_history": self._train_history,
        }


# Singleton
_hedging_engine: Optional[DynamicHedgingEngine] = None

def get_hedging_engine(agent_type: str = "dqn") -> DynamicHedgingEngine:
    global _hedging_engine
    if _hedging_engine is None or _hedging_engine.agent_type != agent_type:
        _hedging_engine = DynamicHedgingEngine(agent_type=agent_type)
    return _hedging_engine
