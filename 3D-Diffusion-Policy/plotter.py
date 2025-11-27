import matplotlib.pyplot as plt
from IPython import display  # works in Jupyter / notebook
import numpy as np
class JointPlotter:
    def __init__(self, num_joints=6):
        self.num_joints = num_joints
        self.joint_history = []  # list of joint arrays
        self.fig, self.ax = plt.subplots()
        self.lines = [self.ax.plot([], [], label=f'joint {i+1}')[0] for i in range(num_joints)]
        self.ax.set_xlabel('Step')
        self.ax.set_ylabel('Joint Angle (rad)')
        self.ax.set_title('Live Joint Trajectory')
        self.ax.legend()
        plt.ion()
        plt.show()

    def update(self, joints):
        """
        joints: np.array of shape [num_joints]
        """
        self.joint_history.append(joints.copy())
        history = np.array(self.joint_history)  # [steps, num_joints]
        steps = np.arange(len(self.joint_history))

        for i in range(self.num_joints):
            self.lines[i].set_data(steps, history[:, i])

        self.ax.relim()
        self.ax.autoscale_view()
        display.display(self.fig)
        display.clear_output(wait=True)
        plt.pause(0.001)  # very small pause for live update


class ActionPlot:
    def __init__(self, num_actions, labels=None, title="Predicted Actions"):
        """
        Live plot for DP3-predicted actions.

        Args:
            num_actions (int): Number of action dimensions (e.g., 7: 3 pos + 4 quat).
            labels (list of str): Labels for each action component.
        """
        self.num_actions = num_actions
        self.history = []  # stores all predicted actions [steps, num_actions]

        self.fig, self.ax = plt.subplots()
        labels = labels if labels is not None else [f"Action {i+1}" for i in range(num_actions)]
        self.lines = [self.ax.plot([], [], label=labels[i])[0] for i in range(num_actions)]

        self.ax.set_title(title)
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Action Value")
        self.ax.legend()
        plt.ion()
        plt.show()

    def update(self, action):
        """
        Update plot with new action.

        Args:
            action (np.array or list): Current action [num_actions]
        """
        action = np.array(action).reshape(-1)
        assert action.shape[0] == self.num_actions, f"Expected {self.num_actions} dims, got {action.shape[0]}"
        
        self.history.append(action.copy())
        history = np.array(self.history)
        steps = np.arange(len(self.history))

        for i in range(self.num_actions):
            self.lines[i].set_data(steps, history[:, i])

        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class ActionPosePlot:
    def __init__(self, labels=None, action_dim=7, plot_tcp=True):
        """
        Live plot for actions and TCP pose changes.
        
        Args:
            action_dim (int): Dimension of action (e.g., 6 or 7)
            plot_tcp (bool): If True, also show 3D TCP positions
        """
        self.action_dim = action_dim
        self.action_history = []       # list of actions
        self.tcp_history = []          # list of TCP positions (3D)
        self.plot_tcp = plot_tcp
        
        # Setup figure
        self.fig = plt.figure(figsize=(12,5))
        
        # Left: action plot
        self.ax_action = self.fig.add_subplot(1,2,1)
        labels = labels if labels is not None else [f"Action {i+1}" for i in range(action_dim)]
        self.lines = [self.ax_action.plot([], [], label=labels[i])[0] for i in range(action_dim)]
        self.ax_action.set_xlabel("Step")
        self.ax_action.set_ylabel("Action Value")
        self.ax_action.set_title("DP3 Actions")
        self.ax_action.legend()
        
        # Right: TCP 3D plot
        if plot_tcp:
            self.ax_tcp = self.fig.add_subplot(1,2,2, projection='3d')
            self.ax_tcp.set_title("TCP Position")
            self.ax_tcp.set_xlabel("X")
            self.ax_tcp.set_ylabel("Y")
            self.ax_tcp.set_zlabel("Z")
            # self.tcp_line, = self.ax_tcp.plot([], [], [], c='blue', marker='o', label='TCP')
            self.tcp_line, = self.ax_tcp.plot([], [], [], 'b-', lw=1, label='TCP Trajectory')
            # Small marker for current position
            self.tcp_dot, = self.ax_tcp.plot([], [], [], 'ro', markersize=3, label='Current TCP')
        
        
        plt.ion()
        plt.show()

    def update(self, action, current_tcp):
        """
        Update plot with new action and TCP position.
        
        Args:
            action (np.array): Current action [action_dim]
            current_tcp (np.array): Current TCP position [3]
        """
        action = np.array(action).reshape(-1)
        current_tcp = np.array(current_tcp).reshape(-1)
        
        assert action.shape[0] == self.action_dim, f"Expected {self.action_dim}, got {action.shape[0]}"
        assert current_tcp.shape[0] == 3, "TCP position must be 3D"
        
        # --- Update action history ---
        self.action_history.append(action.copy())
        history = np.array(self.action_history)
        steps = np.arange(len(self.action_history))
        for i in range(self.action_dim):
            self.lines[i].set_data(steps, history[:, i])
        self.ax_action.relim()
        self.ax_action.autoscale_view()
        
        # --- Update TCP history ---
        if self.plot_tcp:
            self.tcp_history.append(current_tcp.copy())
            tcp_hist = np.array(self.tcp_history)
            self.tcp_line.set_data(tcp_hist[:,0], tcp_hist[:,1])
            self.tcp_line.set_3d_properties(tcp_hist[:,2])
            self.ax_tcp.relim()
            self.ax_tcp.autoscale_view()
        
        plt.draw()
        plt.pause(0.001)
