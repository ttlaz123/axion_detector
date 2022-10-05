import numpy as np
import matplotlib.pyplot as plt

def load_data(save_path, fname, keep_fails=False):
    fullname = f"{save_path}/{fname}"

    din = np.load(fullname)

    init_poss = din[:,0:6].reshape(-1,6)
    aligned_poss = din[:,6:12].reshape(-1,6)
    aligned_freqs = din[:,12]
    aligned_freqs_err = din[:,13]

    print(f"loaded {save_path}/{fname} with {init_poss.shape[0]} aligns")

    # if the run ended early there will be zeros
    end_inds = np.where(aligned_freqs == 0)[0]
    if len(end_inds) > 0:
        end_ind = end_inds[0]
        print(f"Run ended early, after {end_ind} aligns!")
        init_poss = init_poss[:end_ind]
        aligned_poss = aligned_poss[:end_ind]
        aligned_freqs = aligned_freqs[:end_ind]
        aligned_freqs_err = aligned_freqs_err[:end_ind]

    if not keep_fails:
        # if an autoalign ended up somewhere where a resonance couldn't be fit to, or it reported max iters.
        nofit_inds = np.where(aligned_freqs < 0)[0]
        if len(nofit_inds) > 0:
            print(f"Some ({len(nofit_inds)}) aligns failed, here: {nofit_inds}")
        init_poss = np.delete(init_poss, nofit_inds, axis=0)
        aligned_poss = np.delete(aligned_poss, nofit_inds, axis=0)
        aligned_freqs = np.delete(aligned_freqs, nofit_inds)
        aligned_freqs_err = np.delete(aligned_freqs_err, nofit_inds)

    return init_poss, aligned_poss, aligned_freqs, aligned_freqs_err

if __name__=="__main__":
	save_path="autoalign_hist_data"
	fname_dout = "autoalign_hist_20220919_143431.npy"
	fname_din = 'autoalign_hist_20220916_171219.npy'

	init_poss_o, aligned_poss_o, aligned_freqs_o, aligned_freqs_err_o = load_data(save_path,fname_dout, keep_fails=False)
	init_poss_i, aligned_poss_i, aligned_freqs_i, aligned_freqs_err_i = load_data(save_path,fname_din, keep_fails=False)

	init_poss_o = init_poss_o.reshape(-1,6)
	init_poss_i = init_poss_i.reshape(-1,6)

	'''
	# FREQ HISTOGRAM COMPARISON
	bins = np.linspace(7.52944,7.53168,30)
	plt.hist(aligned_freqs_o*1e-9, bins=bins, alpha=0.5, label="Disks Out")
	plt.hist(aligned_freqs_i*1e-9, bins=bins, alpha=0.5, label="Disks In")
	plt.axvline(np.median(aligned_freqs_o*1e-9), c='b', ls='--', label="Disks Out Median")
	plt.axvline(np.median(aligned_freqs_i*1e-9), c='orange', ls='--', label="Disks In Median")
	plt.title("Aligned Resonant Frequency")
	plt.ylabel("Count")
	plt.xlabel("Frequency (GHz)")
	plt.legend()

	print(np.median(aligned_freqs_o)-np.median(aligned_freqs_i))
	'''
	plt.figure()
	coords = ['X', 'Y', 'Z', 'U', 'V', 'W']
	n = np.arange(init_poss_o.shape[0])
	ind = 88
	for i,coord in enumerate(coords):
	    plt.subplot(231+i)
	    plt.scatter(init_poss_o[:,i], aligned_poss_o[:,i], c=n)
	    plt.scatter(init_poss_o[ind,i], aligned_poss_o[ind,i], c='red')
	    plt.title(f"Alignment Scatter in {coord}")
	    plt.ylabel("Aligned Position (hexa coords)")
	    plt.xlabel("Initial Position (hexa coords)")

	plt.figure()
	plt.scatter(n, aligned_freqs_err_o, c=n)
	plt.scatter(n[ind], aligned_freqs_err_o[ind], c='red')
	plt.title("Error in Fit to Aligned Frequency")

	plt.show()
