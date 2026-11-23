import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

output = "/Users/coeishikawa/Desktop/UROP/tSNE_affinity_beta_explained.pdf"

fig = plt.figure(figsize=(11.7, 8.3), dpi=180)
fig.patch.set_facecolor("#F7F9FC")

# Header
fig.text(0.055, 0.955, "t-SNE Original Affinities: β, σ, and Perplexity",
         fontsize=21, fontweight="bold", color="#162447", va="top")
fig.text(0.055, 0.916,
         "The whole process adapts one Gaussian width for every image i, independently.",
         fontsize=11.5, color="#475569", va="top")

# Helper functions
def box(x, y, w, h, title, body, face="#FFFFFF", edge="#CBD5E1", title_color="#162447"):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.012,rounding_size=0.012",
                           linewidth=1.2, edgecolor=edge, facecolor=face,
                           transform=fig.transFigure)
    fig.patches.append(patch)
    fig.text(x + 0.018, y + h - 0.035, title,
             fontsize=12.5, fontweight="bold", color=title_color,
             transform=fig.transFigure, va="top")
    fig.text(x + 0.018, y + h - 0.087, body,
             fontsize=9.2, color="#334155", linespacing=1.45,
             transform=fig.transFigure, va="top")

def arrow(x1, y1, x2, y2, text="", color="#64748B"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=14,
                        linewidth=1.5, color=color,
                        transform=fig.transFigure)
    fig.patches.append(a)
    if text:
        fig.text((x1+x2)/2, (y1+y2)/2 + 0.018, text,
                 fontsize=8.5, color=color, ha="center",
                 transform=fig.transFigure)

# Top row: input variables
box(0.055, 0.705, 0.270, 0.158,
    "1. Fixed input",
    "For image i:\n• row = squared distances dᵢⱼ to other images j\n• perplexity = desired neighborhood breadth",
    face="#EEF4FF", edge="#93B4E7")

box(0.365, 0.705, 0.270, 0.158,
    "2. Trial Gaussian width",
    "Start β = 1.0\nσ = √(1 / (2β))\nβ is precision; σ is the Gaussian scale.",
    face="#FFF7ED", edge="#FDBA74")

box(0.675, 0.705, 0.270, 0.158,
    "3. Desired entropy",
    "H_target = log(perplexity)\nExample: perplexity 20 → H_target ≈ 2.996",
    face="#ECFDF5", edge="#86EFAC")

# Main equations
box(0.055, 0.482, 0.890, 0.175,
    "Four quantities are linked, but they are not the same",
    "β = 1 / (2σ²)                                      wᵢⱼ = exp[−β(dᵢⱼ − d_min)]\n"
    "σ = √(1 / (2β))                                  p(j|i) = wᵢⱼ / Σₖ wᵢₖ\n"
    "Perplexity = exp(H_i),  H_i = −Σⱼ p(j|i) log p(j|i)",
    face="#FFFFFF", edge="#94A3B8")

# Central iteration flow
box(0.055, 0.235, 0.210, 0.205,
    "A. Trial β",
    "Choose a β value\nand update σ:\nσ = √(1/(2β))",
    face="#FFF7ED", edge="#FDBA74")

box(0.295, 0.235, 0.210, 0.205,
    "B. Calculate weights",
    "w = exp(−β(d−dmin))\nSubtracting dmin\nmakes the nearest weight 1.",
    face="#EEF4FF", edge="#93B4E7")

box(0.535, 0.235, 0.210, 0.205,
    "C. Normalize weights",
    "p(j|i) = wᵢⱼ / Σₖ wᵢₖ\nEach row sums to 1.",
    face="#F5F3FF", edge="#C4B5FD")

box(0.775, 0.235, 0.170, 0.205,
    "D. Check entropy",
    "H_i = −Σ p log p\nCompare H_i\nwith log(perplexity)",
    face="#ECFDF5", edge="#86EFAC")

arrow(0.265, 0.337, 0.295, 0.337)
arrow(0.505, 0.337, 0.535, 0.337)
arrow(0.745, 0.337, 0.775, 0.337)

# Return / bisection feedback
bisection_y = 0.125
box(0.055, bisection_y, 0.890, 0.070,
    "E. Adjust β by bisection",
    "If H_i > H_target: entropy is too spread out → lower β, so increase β.\n"
    "If H_i < H_target: entropy is too concentrated → raise β, so decrease β.\n"
    "Repeat A→D until |H_i−H_target| is below the entropy threshold.",
    face="#FEF2F2", edge="#FCA5A5")

# Arrow feedback on right
feedback = FancyArrowPatch((0.945, bisection_y + 0.035), (0.945, 0.334),
                           connectionstyle="angle3,angleA=90,angleB=0",
                           arrowstyle="-|>", mutation_scale=14,
                           linewidth=1.4, color="#DC2626",
                           transform=fig.transFigure)
fig.patches.append(feedback)
fig.text(0.952, 0.205, "repeat",
         fontsize=8.2, color="#DC2626", rotation=90,
         transform=fig.transFigure, va="center")

# Bottom key takeaway
fig.text(0.055, 0.085,
         "Important: β is only a trial parameter here. The search does not keep changing σ directly; it computes σ from the current trial β.",
         fontsize=10.1, fontweight="bold", color="#9F1239", va="top")
fig.text(0.055, 0.050,
         "Once β is accepted, store σᵢ = √(1/(2βᵢ)), the final probabilities p(j|i), and achieved perplexity exp(Hᵢ).",
         fontsize=9.6, color="#475569", va="top")

plt.subplots_adjust(left=0.05, right=0.95, bottom=0.04, top=0.94)
from matplotlib.backends.backend_pdf import PdfPages
with PdfPages(output) as pdf:
    pdf.savefig(fig, bbox_inches=None)

print(output)
