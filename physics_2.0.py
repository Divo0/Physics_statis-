import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

#calculate share force and momentum on beam, program


class DistributedLoad:
    def __init__(self, func_str, start, end):
        self.func_str = func_str
        self.func = lambda x: eval(func_str)
        self.start = start
        self.end = end
        self.total_force, _ = quad(self.func, start, end)
        self.centroid = quad(lambda x: x * self.func(x), start, end)[0] / self.total_force


def plot_beam(ax, L, point_loads, dist_loads, show_reactions=False):
    ax.clear()
    # Draw beam
    ax.plot([0, L], [0, 0], 'k-', linewidth=4, zorder=1)

    # Draw supports
    ax.plot(0, 0, '^', markersize=15, color='k', zorder=2)
    ax.plot(L, 0, '^', markersize=15, color='k', zorder=2)

    # Draw point forces
    for x, F in point_loads:
        color = 'blue' if F > 0 else 'red'
        direction = 1 if F > 0 else -1
        arrow_length = L / 10
        ax.arrow(x, 0, 0, direction * arrow_length,
                 head_width=L / 30, head_length=arrow_length / 2,
                 fc=color, ec=color, zorder=3)
        ax.text(x + L / 50, direction * (arrow_length + L / 15),
                f'{abs(F):.1f} N', color=color,
                ha='left' if F > 0 else 'right', va='center')

    # Draw distributed loads with actual function shape
    for dl in dist_loads:
        x = np.linspace(dl.start, dl.end, 100)
        y = [dl.func(xi) for xi in x]
        ax.fill_between(x, y, alpha=0.3, color='green')
        ax.plot(x, y, 'g-', linewidth=2)

        # Annotation with function and total force
        max_y = max(y) if len(y) > 0 else 0
        ax.text((dl.start + dl.end) / 2, max_y + L / 20,
                f'q(x) = {dl.func_str}\nTotal: {dl.total_force:.1f} N',
                ha='center', color='darkgreen', fontsize=8)

    ax.set_xlim(-L / 5, L + L / 5)
    ax.set_ylim(-L / 2, L / 2)
    ax.set_title("Beam Loading Diagram" + (" (With Reactions)" if show_reactions else ""))
    ax.set_xlabel("Position (m)")
    ax.get_yaxis().set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.7)


def calculate_diagrams(L, point_loads, dist_loads, reactions):
    x_vals = np.linspace(0, L, 1000)
    shear = np.zeros_like(x_vals)
    moment = np.zeros_like(x_vals)

    all_loads = point_loads + reactions

    for i, x in enumerate(x_vals):
        # Point loads contribution
        shear[i] = sum(F for xf, F in all_loads if xf <= x)
        moment[i] = sum(F * (x - xf) for xf, F in all_loads if xf <= x)

        # Distributed loads contribution
        for dl in dist_loads:
            if x >= dl.start:
                a = max(dl.start, 0)
                b = min(dl.end, x)
                if b > a:
                    shear[i] += quad(dl.func, a, b)[0]
                    moment[i] += quad(lambda xi: (x - xi) * dl.func(xi), a, b)[0]

    return x_vals, shear, moment


def plot_diagrams(x_vals, shear, moment):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    ax1, ax2 = axes  # Unpack the axes

    # Shear diagram
    ax1.plot(x_vals, shear, 'r-', linewidth=2)
    ax1.fill_between(x_vals, shear, color='red', alpha=0.3)

    # Find and annotate shear extremes
    shear_min_idx = np.argmin(shear)
    shear_max_idx = np.argmax(shear)
    for idx, prefix in [(shear_min_idx, 'Min'), (shear_max_idx, 'Max')]:
        ax1.plot(x_vals[idx], shear[idx], 'ko')
        ax1.annotate(f'{prefix}: {shear[idx]:.1f} N\n@ {x_vals[idx]:.1f} m',
                     (x_vals[idx], shear[idx]),
                     textcoords="offset points", xytext=(0, 10 if prefix == 'Min' else -15),
                     ha='center', va='center')

    ax1.set_title("Shear Force Diagram")
    ax1.set_ylabel("Shear Force (N)")
    ax1.grid(True)
    ax1.axhline(0, color='k', linewidth=0.5)

    # Moment diagram
    ax2.plot(x_vals, moment, 'b-', linewidth=2)
    ax2.fill_between(x_vals, moment, color='blue', alpha=0.3)

    # Find and annotate moment extremes
    moment_min_idx = np.argmin(moment)
    moment_max_idx = np.argmax(moment)
    for idx, prefix in [(moment_min_idx, 'Min'), (moment_max_idx, 'Max')]:
        ax2.plot(x_vals[idx], moment[idx], 'ko')
        ax2.annotate(f'{prefix}: {moment[idx]:.1f} Nm\n@ {x_vals[idx]:.1f} m',
                     (x_vals[idx], moment[idx]),
                     textcoords="offset points", xytext=(0, 10 if prefix == 'Min' else -15),
                     ha='center', va='center')

    ax2.set_title("Bending Moment Diagram")
    ax2.set_xlabel("Position along beam (m)")
    ax2.set_ylabel("Bending Moment (Nm)")
    ax2.grid(True)
    ax2.axhline(0, color='k', linewidth=0.5)

    plt.tight_layout()


def main():
    # Input parameters
    L = float(input("Enter beam length (m): "))

    # Point loads input
    point_loads = []
    n_points = int(input("Enter number of point loads: "))
    for i in range(n_points):
        x = float(input(f"\nPoint load {i + 1} position (0-{L}): "))
        F = float(input(f"Point load {i + 1} value (N, +↑/-↓): "))
        point_loads.append((x, F))

    # Distributed loads input
    dist_loads = []
    n_dist = int(input("\nEnter number of distributed loads: "))
    for i in range(n_dist):
        func_str = input(f"Distributed load {i + 1} function (e.g., '100', '50*x', '20*(x-2)**2'): ")
        start = float(input(f"Start position (0-{L}): "))
        end = float(input(f"End position ({start}-{L}): "))
        dist_loads.append(DistributedLoad(func_str, start, end))

    # Calculate reactions
    sum_F = sum(F for _, F in point_loads) + sum(dl.total_force for dl in dist_loads)
    sum_M = sum(F * x for x, F in point_loads) + sum(dl.total_force * dl.centroid for dl in dist_loads)
    R2 = -sum_M / L
    R1 = -sum_F - R2
    reactions = [(0, R1), (L, R2)]

    print(f"\nReactions: R₁ = {R1:.2f} N, R₂ = {R2:.2f} N")

    # Plot beam with all loads
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_beam(ax, L, point_loads + reactions, dist_loads, show_reactions=True)

    # Calculate and plot diagrams
    x_vals, shear, moment = calculate_diagrams(L, point_loads, dist_loads, reactions)
    plot_diagrams(x_vals, shear, moment)

    plt.show()  # Show all figures at once here


if __name__ == "__main__":
    main()