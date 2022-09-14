import numpy as np
import matplotlib.pyplot as plt

def load_data(save_path, fname, keep_fails=False):
    fullname = f"{save_path}/{fname}"

    din = np.load(fullname)

    init_poss = din[:,0:6]
    aligned_poss = din[:,6:12]
    aligned_freqs = din[:,12]
    aligned_freqs_err = din[:,13]

    # if the run ended early there will be zeros
    end_inds = np.where(aligned_freqs == 0)[0]
    if len(end_inds) > 0:
        end_ind = end_inds[0]
        print(f"Run ended early, after {end_ind}/{init_poss.shape[0]} autoaligns!")
        init_poss = init_poss[:end_ind]
        aligned_poss = aligned_poss[:end_ind]
        aligned_freqs = aligned_freqs[:end_ind]
        aligned_freqs_err = aligned_freqs_err[:end_ind]

    if not keep_fails:
        # if an autoalign ended up somewhere where a resonance couldn't be fit to, or it reported max iters.
        nofit_inds = np.where(aligned_freqs < 0)[0]
        if len(nofit_inds) > 0:
            print(f"Some ({len(nofit_inds)}) aligns failed, here: {nofit_inds}")
        init_poss = np.delete(init_poss, nofit_inds)
        aligned_poss = np.delete(aligned_poss, nofit_inds)
        aligned_freqs = np.delete(aligned_freqs, nofit_inds)
        aligned_freqs_err = np.delete(aligned_freqs_err, nofit_inds)

    return init_poss, aligned_poss, aligned_freqs, aligned_freqs_err

init_poss, aligned_poss, aligned_freqs, aligned_freqs_err = load_data("autoalign_hist_data", "autoalign_hist_20220913_143118.npy")

plt.hist(aligned_freqs)
plt.show()
