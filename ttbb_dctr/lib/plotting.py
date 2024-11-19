import os
import yaml
import numpy as np
import awkward as ak
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import mplhep as hep
import hist
from hist import Hist

from ttbb_dctr.lib.data_preprocessing import get_input_features

np.seterr(divide="ignore", invalid="ignore", over="ignore")

matplotlib.use("Agg")
hep.style.use("CMS")
plt.rcParams["figure.figsize"] = [8,8]
plt.rcParams["font.size"] = 18
CMAP_6 = plt.rcParams['axes.prop_cycle'].by_key()['color']

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

def plot_correlation(x, y, w, varname_x, title, score, plot_dir):
    print(f"Plotting {score} vs {varname_x} for {title}")
    bins_feature = bins[varname_x]
    ranges_feature = ranges[varname_x]
    x, y, w = np.array(x), np.array(y), np.array(w)
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
    plt.close(fig)

def plot_correlation_matrix(df, title, plot_dir, suffix=None):
    corr = df.corr()
    fig, ax = plt.subplots(1,1, figsize=(16,16))
    sns.heatmap(corr, annot=False, cmap='coolwarm', square=True, linewidths=0.5, vmin=-1)

    ax.set_title(f"Correlation Matrix for {title}", fontsize=24)

    filename = os.path.join(plot_dir, f"correlation_matrix_{title.lower()}.png")
    if suffix is not None:
        filename = filename.replace(".png", f"_{suffix}.png")
    print(f"Saving {filename}")
    plt.savefig(filename, dpi=300)
    plt.close(fig)

def plot_correlation_matrix_diff(df1, df2, title, plot_dir, suffix=None):
    corr1 = df1.corr()
    corr2 = df2.corr()
    corr_diff = corr2 - corr1
    fig, ax = plt.subplots(1,1, figsize=(16,16))
    sns.heatmap(corr_diff, annot=False, cmap='coolwarm', square=True, linewidths=0.5, vmin=-1, vmax=1)

    ax.set_title(f"Correlation Matrix Difference (CR2 - CR1) for {title}", fontsize=24)

    filename = os.path.join(plot_dir, f"correlation_matrix_diff_{title.lower()}.png")
    if suffix is not None:
        filename = filename.replace(".png", f"_{suffix}.png")
    print(f"Saving {filename}")
    plt.savefig(filename, dpi=300)
    plt.close(fig)

def plot_single_classifier_score(events, mask_data, mask_ttbb, plot_dir, suffix=None):
    nbins=200

    score = events.dctr_score
    fig, ax = plt.subplots(1,1,figsize=[8,8])
    ax.hist(score[mask_ttbb], weights=events.event.weight[mask_ttbb], histtype="step", bins=nbins, range=(0,1), label="ttbb")
    ax.hist(score[mask_data], weights=events.event.weight[mask_data], histtype="step", bins=nbins, range=(0,1), label="Data - minor bkg")
    ax.set_xlabel("Classifier score")
    ax.legend()
    filename = os.path.join(plot_dir, "score_classifier.png")
    if suffix is not None:
        filename = filename.replace(".png", f"_{suffix}.png")
    print(f"Saving {filename}")
    plt.savefig(filename, dpi=300)
    plt.close(fig)

def plot_classifier_score(events, mask_data, mask_ttbb, mask_train, plot_dir):
    print("Plotting classifier score")
    for mask, label in zip([mask_train, ~mask_train], ["Training", "Validation"]):
        fig, ax = plt.subplots(1,1,figsize=[8,8])
        plot_single_classifier_score(events[mask], mask_data[mask], mask_ttbb[mask], plot_dir, suffix=label.lower())

def plot_single_dctr_weight(events, mask, ax, plot_dir, stack, suffix=None):
    nbins = 50
    axis_w = hist.axis.Regular(nbins, 0, 2.5, flow=False, name="w")
    axis_cat = hist.axis.StrCategory(["njet=4", "njet=5", "njet=6", "njet>=7"], name="njet")
    full_hist = Hist(axis_w, axis_cat)

    mask_ttbb = events.ttbb == 1
    weight_ttbb = events.dctr_weight
    weight_ttbb = weight_ttbb[mask_ttbb]
    events = events[mask_ttbb]
    njet = ak.num(events.JetGood)
    mask = mask[mask_ttbb]

    for nj in [4,5,6,7]:
        if nj == 7:
            mask_nj = (njet >= nj)
            njet_label = f"njet>={nj}"
        else:
            mask_nj = (njet == nj)
            njet_label = f"njet={nj}"
        full_hist.fill(w=weight_ttbb[mask & mask_nj], weight=events.event.weight[mask & mask_nj], njet=njet_label)
        s = full_hist.stack("njet")
    w = np.array(events.event.weight[mask])
    ax.hist(weight_ttbb[mask], weights=w, histtype="step", label="ttbb", bins=nbins, range=(0,2.5), linewidth=2, color="black")

    if stack:
        s.plot(stack=stack, histtype="fill", ax=ax)
    else:
        s.plot(stack=stack, histtype="step", ax=ax)
    ax.set_xlabel("Weight $\omega = p(Data - bkg) / p(ttbb)$")
    ax.set_ylabel("Counts")
    ax.legend()
    filename = os.path.join(plot_dir, "weight_ttbb.png")
    if suffix is not None:
        filename = filename.replace(".png", f"_{suffix}.png")
    if stack:
        filename = filename.replace(".png", "_stack.png")
    print(f"Saving {filename}")
    plt.savefig(filename, dpi=300)

def plot_dctr_weight(events, mask_train, plot_dir):
    print("Plotting DCTR weight")
    for stack in [True, False]:
        #fig, axes = plt.subplots(1,2,figsize=[16,8])
        #axes = axes.flatten()
        for i, (mask, label) in enumerate(zip([mask_train, ~mask_train], ["Training", "Validation"])):
            fig, ax = plt.subplots(1,1,figsize=[8,8])
            plot_single_dctr_weight(events, mask, ax, plot_dir, stack, suffix=label.lower())
            ax.set_title(label)
            plt.close(fig)

        #filename = os.path.join(plot_dir, "weight_ttbb_train_test.png")
        #if stack:
        #    filename = filename.replace(".png", "_stack.png")
        #plt.savefig(filename, dpi=300)
        #plt.close(fig)

def get_central_interval(x, loc=1.0,  perc=0.68):
    """Get the interval of data around a specified value."""
    x_below = x[x < loc]
    x_above = x[x >= loc]
    x_lo = np.quantile(x_below, 1 - perc)
    x_hi = np.quantile(x_above, perc)
    return x_lo, x_hi

def get_quantiles_by_njet(events, mask, q=[1./3, 2./3]):
    njet = ak.num(events.JetGood)
    w_dctr = events.dctr_weight
    weight_cuts_dict = {}
    for nj in range(4,7):
        mask_nj = (njet == nj)
        w_lo, w_hi = np.quantile(w_dctr[mask & mask_nj].to_numpy(), q)
        weight_cuts_dict[f"njet={nj}"] = [(0, w_lo), (w_lo, w_hi), (w_hi, 3)]

    w_lo, w_hi = np.quantile(w_dctr[mask & (njet >= 6)].to_numpy(), q)
    weight_cuts_dict["njet>=6"] = [(0, w_lo), (w_lo, w_hi), (w_hi, 3)]
    w_lo, w_hi = np.quantile(w_dctr[mask & (njet >= 7)].to_numpy(), q)
    weight_cuts_dict["njet>=7"] = [(0, w_lo), (w_lo, w_hi), (w_hi, 3)]

    return weight_cuts_dict

def get_bin_centers(bins):
    binwidth = np.ediff1d(bins)
    bin_centers = bins[:-1] + 0.5*binwidth
    return bin_centers

def plot_closure_test(events, mask_data, mask_ttbb, plot_dir, only_var=None, density=False):
    input_features = get_input_features(events)
    weight_ttbb = events.dctr_weight
    weights = events.event.weight
    assert type(input_features) == dict, "Input features should be a dictionary."
    for varname, x in input_features.items():
        if only_var is not None and varname != only_var: continue
        fig, (ax, rax) = plt.subplots(2,1,figsize=[8,8], gridspec_kw={"height_ratios" : [3,1]}, sharex=True)
        x = np.array(x)
        w = np.array(weights)
        h = ax.hist(x[mask_data], weights=w[mask_data], bins=bins[varname], range=ranges[varname], histtype="step", label="data - minor bkg", color="black", linewidth=3, density=density)
        sumw2 = np.histogram(x[mask_data], bins=bins[varname], range=ranges[varname], weights=w[mask_data]**2)[0]
        yerr = np.sqrt(sumw2)
        # Plot the statistical uncertainty on top of the histogram with errorbars
        bin_centers = get_bin_centers(h[1])
        ax.errorbar(bin_centers, h[0], yerr=yerr, fmt='none', color='black', capsize=5, capthick=2)
        try:
            weights_original = events.event.weight_original
            w_original = np.array(weights_original)
            h_ttbb = ax.hist(x[mask_ttbb], weights=w_original[mask_ttbb], bins=bins[varname], range=ranges[varname], color=CMAP_6[0], histtype="step", label="ttbb", linewidth=2, density=density)
        except:
            print("WARNING: the field 'event.weight_original' is not available. Skip plotting the original ttbb distribution.")
        h_ttbb_rwg_1d = ax.hist(x[mask_ttbb], weights=w[mask_ttbb], bins=bins[varname], range=ranges[varname], color=CMAP_6[1], histtype="step", label="ttbb (1D rwg.)", linewidth=2, density=density)
        h_ttbb_rwg_dctr = ax.hist(x[mask_ttbb], weights=w[mask_ttbb]*weight_ttbb[mask_ttbb], bins=bins[varname], range=ranges[varname], color=CMAP_6[2], histtype="step", label="ttbb (DNN rwg.)", linewidth=2, density=density)

        rax.hlines(1.0, *ranges[varname], colors='gray', linestyles='dashed')
        binwidth = np.ediff1d(h[1])
        try:
            #rax.stairs(h_ttbb[0] / h[0], h[1], color=CMAP_6[0], linewidth=2)
            rax.errorbar(bin_centers, h_ttbb[0] / h[0], xerr=0.5*binwidth, yerr=0, fmt='none', color=CMAP_6[0], linewidth=2)
        except:
            pass
        #rax.stairs(h_ttbb_rwg_1d[0] / h[0], h[1], color=CMAP_6[1], linewidth=2)
        #rax.stairs(h_ttbb_rwg_dctr[0] / h[0], h[1], color=CMAP_6[2], linewidth=2)
        binwidth = np.ediff1d(h[1])
        rax.errorbar(bin_centers, h_ttbb_rwg_1d[0] / h[0], xerr=0.5*binwidth, yerr=0, fmt='none', color=CMAP_6[1], linewidth=2)
        rax.errorbar(bin_centers, h_ttbb_rwg_dctr[0] / h[0], xerr=0.5*binwidth, yerr=0, fmt='none', color=CMAP_6[2], linewidth=2)

        # Plot errorband in ratio plot for data - minor bkg, centered in 1 with a width of 1 sigma with respect to the statistical uncertainty
        unc_down = 1 - yerr / h[0]
        unc_up = 1 + yerr / h[0]
        unc_band = np.array([unc_down, unc_up])
        rax.fill_between(h[1], np.r_[unc_band[0], unc_band[0][-1]], np.r_[unc_band[1], unc_band[1][-1]], label="stat. unc.",
                            color=[0.,0.,0.,0.4], facecolor=[0.,0.,0.,0.], hatch="////", linewidth=0, step="post")

        ax.set_xlim(*ranges[varname])
        ax.set_ylim(0, 1.4*max(h[0]))
        ax.set_ylabel("Counts")

        rax.set_xlabel(varname)
        rax.set_ylabel("ttbb / (Data - minor bkg)", fontsize=10)
        rax.set_ylim(0,2)
        ax.legend()
        #rax.legend()
        filename = os.path.join(plot_dir, f"{varname}_reweighed.png")
        print(f"Saving {filename}")
        plt.savefig(filename, dpi=300)
        plt.close(fig)


def plot_closure_test_split_by_weight(events, mask_data, mask_ttbb, weight_cuts, plot_dir, only_var=None, density=False, suffix=None):
    # Assert that weight_cuts is an iterable
    input_features = get_input_features(events)
    weight_ttbb = events.dctr_weight
    weights = events.event.weight
    assert hasattr(weight_cuts, "__iter__"), "weight_cuts should be an iterable."
    assert len(weight_cuts[0]) == 2, "weight_cuts should contain tuples of two elements: `(w_lo, w_hi)`."
    ncuts = len(weight_cuts)
    height = 8

    for varname, x in input_features.items():
        if only_var is not None and varname != only_var: continue
        fig, axes = plt.subplots(2,ncuts,figsize=[(ncuts+0.2)*height,height], gridspec_kw={"height_ratios" : [3,1]}, sharex=True)
        axes = axes.flatten()
        for j, (weight_lo, weight_hi) in enumerate(weight_cuts):
            mask_weight = (weight_ttbb >= weight_lo) & (weight_ttbb < weight_hi)
            ax = axes[j]
            rax = axes[ncuts+j]
            x = np.array(x)
            w = np.array(weights)
            h = ax.hist(x[mask_data & mask_weight], weights=w[mask_data & mask_weight], bins=bins[varname], range=ranges[varname], histtype="step", label="data - minor bkg", color="black", linewidth=3, density=density)
            sumw2 = np.histogram(x[mask_data & mask_weight], bins=bins[varname], range=ranges[varname], weights=w[mask_data & mask_weight]**2)[0]
            yerr = np.sqrt(sumw2)
            # Plot the statistical uncertainty on top of the histogram with errorbars
            bin_centers = get_bin_centers(h[1])
            ax.errorbar(bin_centers, h[0], yerr=yerr, fmt='none', color='black', capsize=5, capthick=2)
            try:
                weights_original = events.event.weight_original
                w_original = np.array(weights_original)
                h_ttbb = ax.hist(x[mask_ttbb & mask_weight], weights=w_original[mask_ttbb & mask_weight], bins=bins[varname], range=ranges[varname], color=CMAP_6[0], histtype="step", label="ttbb", linewidth=2, density=density)
            except:
                print("WARNING: the field 'event.weight_original' is not available. Skip plotting the original ttbb distribution.")
            h_ttbb_rwg_1d = ax.hist(x[mask_ttbb & mask_weight], weights=w[mask_ttbb & mask_weight], bins=bins[varname], range=ranges[varname], color=CMAP_6[1], histtype="step", label="ttbb (1D rwg.)", linewidth=2, density=density)
            h_ttbb_rwg_dctr = ax.hist(x[mask_ttbb & mask_weight], weights=w[mask_ttbb & mask_weight]*weight_ttbb[mask_ttbb & mask_weight], bins=bins[varname], range=ranges[varname], color=CMAP_6[2], histtype="step", label="ttbb (DNN rwg.)", linewidth=2, density=density)
            rax.hlines(1.0, *ranges[varname], colors='gray', linestyles='dashed')
            binwidth = np.ediff1d(h[1])
            try:
                #rax.stairs(h_ttbb[0] / h[0], h[1], color=CMAP_6[0], linewidth=2)
                rax.errorbar(bin_centers, h_ttbb[0] / h[0], xerr=0.5*binwidth, yerr=0, fmt='none', color=CMAP_6[0], linewidth=2)
            except:
                pass
            #rax.stairs(h_ttbb_rwg_1d[0] / h[0], h[1], color=CMAP_6[1], linewidth=2)
            #rax.stairs(h_ttbb_rwg_dctr[0] / h[0], h[1], color=CMAP_6[2], linewidth=2)
            rax.errorbar(bin_centers, h_ttbb_rwg_1d[0] / h[0], xerr=0.5*binwidth, yerr=0, fmt='none', color=CMAP_6[1], linewidth=2)
            rax.errorbar(bin_centers, h_ttbb_rwg_dctr[0] / h[0], xerr=0.5*binwidth, yerr=0, fmt='none', color=CMAP_6[2], linewidth=2)
            # Plot errorband in ratio plot for data - minor bkg, centered in 1 with a width of 1 sigma with respect to the statistical uncertainty
            unc_down = 1 - yerr / h[0]
            unc_up = 1 + yerr / h[0]
            unc_band = np.array([unc_down, unc_up])
            rax.fill_between(h[1], np.r_[unc_band[0], unc_band[0][-1]], np.r_[unc_band[1], unc_band[1][-1]], label="stat. unc.",
                             color=[0.,0.,0.,0.4], facecolor=[0.,0.,0.,0.], hatch="////", linewidth=0, step="post")
            #rax.fill_between(h[1][:-1], 1 - yerr / h[0], 1 + yerr / h[0], color='gray', label="stat. unc.")
            ax.set_xlim(*ranges[varname])
            ax.set_ylim(0, 1.4*max(max(h[0]), max(h_ttbb_rwg_dctr[0]), max(h_ttbb_rwg_1d[0])))
            ax.set_title(f"$\omega\in$[{round(weight_lo, 2)},{round(weight_hi, 2)})")
            ax.set_ylabel("Counts")
            rax.set_xlabel(varname)
            rax.set_ylabel("ttbb / (Data - minor bkg)", fontsize=10)
            rax.set_ylim(0,2)
            ax.legend()
            #rax.legend()
        if suffix is not None:
            if not os.path.exists(os.path.join(plot_dir, suffix)):
                os.makedirs(os.path.join(plot_dir, suffix), exist_ok=True)
            filename = os.path.join(plot_dir, suffix, f"{varname}_reweighed_split_by_weight.png")
        else:
            filename = os.path.join(plot_dir, f"{varname}_reweighed_split_by_weight.png")
        print(f"Saving {filename}")
        plt.savefig(filename, dpi=300)
        plt.close(fig)

def plot_reweighting_map(input_file, output_file=None):
    # Check that file is yaml otherwise raise an error
    if not input_file.endswith(".yaml"):
        raise ValueError("Input file should be in yaml format.")
    print(f"Reading reweighting map from {input_file}")
    with open(input_file, "r") as f:
        reweighting_map = yaml.safe_load(f)
    fig, ax = plt.subplots(1,1, figsize=(8,8))

    for mask_name, reweighting_map_mask in reweighting_map.items():
        njet = np.array(list(reweighting_map_mask.keys()))
        reweighting = np.array(list(reweighting_map_mask.values()))
        label = mask_name
        if label.endswith("_ttlf0p30"):
            label = label.replace("_ttlf0p30", ")")
        # Convert numbers in format 0p30To0p60 to ttHbb \in (0.30, 0.60)
        # Expected format of the key: tthbb0p10To0p60_ttlf0p30
        if label.startswith("tthbb"):
            label = label.replace("tthbb", "ttHbb score $\in$ (")
            label = label.replace("0p", "0.")
            label = label.replace("To", ", ")

        ax.plot(njet, reweighting, label=label)

    ax.legend(fontsize=16)
    ax.set_xlabel("$N_{jet}$")
    ax.set_ylabel("SF")
    ax.set_xlim(4, 8)
    ax.set_ylim(1, 1.75)
    ax.set_xticks(np.arange(4, 9))
    # Replace last tick to ">7"
    ax.set_xticklabels([f"{i}" if i < 8 else ">7" for i in range(4, 9)])
    if output_file is None:
        output_file = input_file.replace(".yaml", ".png")
    print("Saving plot to", output_file)
    plt.savefig(output_file, dpi=300)
    ax.set_title("1D reweighting factor", fontsize=18)
    output_with_title = output_file.replace(".png", "_with_title.png")
    print(f"Saving plot to {output_with_title}")
    plt.savefig(output_with_title, dpi=300)
