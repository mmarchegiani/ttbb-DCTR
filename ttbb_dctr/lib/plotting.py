import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("CMS")
plt.rcParams["figure.figsize"] = [8,8]
plt.rcParams["font.size"] = 18



bins = {
    "njet" : 6,
    "nbjet" : 5,
    "ht" : 25,
    "ht_b" : 25,
    "ht_light" : 25,
    "drbb_avg" : 25,
    "mbb_max" : 25,
    "mbb_min" : 25,
    "mbb_closest" : 25,
    "drbb_min" : 25,
    "detabb_min" : 25,
    "dphibb_min" : 25,
    "jet_pt_1" : 25,
    "jet_pt_2" : 25,
    "jet_pt_3" : 25,
    "jet_pt_4" : 25,
    "bjet_pt_1" : 25,
    "bjet_pt_2" : 25,
    "bjet_pt_3" : 25,
    "jet_eta_1" : 25,
    "jet_eta_2" : 25,
    "jet_eta_3" : 25,
    "jet_eta_4" : 25,
    "bjet_eta_1" : 25,
    "bjet_eta_2" : 25,
    "bjet_eta_3" : 25
}

ranges = {
    "njet" : (4,10),
    "nbjet" : (3,8),
    "ht" : (0,1000),
    "ht_b" : (0,1000),
    "ht_light" : (0,1000),
    "drbb_avg" : (0,5),
    "mbb_max" : (0,500),
    "mbb_min" : (0,500),
    "mbb_closest" : (0,500),
    "drbb_min" : (0,5),
    "detabb_min" : (0,5),
    "dphibb_min" : (0,5),
    "jet_pt_1" : (0,500),
    "jet_pt_2" : (0,500),
    "jet_pt_3" : (0,500),
    "jet_pt_4" : (0,500),
    "bjet_pt_1" : (0,500),
    "bjet_pt_2" : (0,500),
    "bjet_pt_3" : (0,500),
    "jet_eta_1" : (-2.4, 2.4),
    "jet_eta_2" : (-2.4, 2.4),
    "jet_eta_3" : (-2.4, 2.4),
    "jet_eta_4" : (-2.4, 2.4),
    "bjet_eta_1" : (-2.4, 2.4),
    "bjet_eta_2" : (-2.4, 2.4),
    "bjet_eta_3" : (-2.4, 2.4)
}

def plot_correlation(x_dict, events, title, score, plot_dir):
    assert len(x_dict) == 1, "Only one variable can be plotted"
    varname_x, x = x_dict.popitem()
    print(f"Plotting {score} vs {varname_x} for {title}")
    bins_feature = bins[varname_x]
    ranges_feature = ranges[varname_x]
    x = np.array(x)
    y = np.array(events.spanet_output[score])
    w = np.array(events.event.weight)
    # Invert negative weights for minor backgrounds
    if title not in ["Data", "ttbb"]:
        w = -w
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    hep.cms.label("Preliminary", loc=0, ax=ax)
    h = ax.hist2d(x, y, weights=w, bins=[bins_feature, 100], range=[ranges_feature, (0, 1)], cmap="viridis", norm=matplotlib.colors.LogNorm(vmin=0.001, vmax=10**3))
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel(varname_x)
    ax.set_ylabel(score)
    #ax.set_title(title)
    filename = os.path.join(plot_dir, f"{title.lower()}_{score}_vs_{varname_x}.png")
    print(f"Saving {filename}")
    plt.savefig(filename, dpi=300)
    plt.close()
