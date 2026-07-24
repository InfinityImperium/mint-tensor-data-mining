import os


# Single-thread OpenMP/MKL before torch/pyscamp import so their bundled
# OpenMP runtimes cannot conflict. (Removing KMP_DUPLICATE_LIB_OK makes a
# duplicate-runtime load fail loudly instead of being papered over.)
os.environ.pop("KMP_DUPLICATE_LIB_OK", None)
# setdefault: the single-thread guard applies unless the environment says
# otherwise (e.g. docker run -e OMP_NUM_THREADS=$(nproc) on Linux, where the
# duplicate-runtime conflict cannot occur and the pin would only throttle).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["KMP_SETTINGS"] = "FALSE"




import numpy as np
import torch
from contextlib import nullcontext
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_to_tensor
from kneed import KneeLocator
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import pyscamp
import warnings






# Set backend for tensorly when needed externally
try:
    tl.set_backend('pytorch')
except Exception:
    pass


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nnrobustpca_stable_pcp(
    M,
    lambda_param=None,
    mu=None,
    max_iter=1000,
    tol=1e-5,
    alpha=1.0,          
    beta=1.0,          
    check_every=1,     # compute residual every N iters
    verbose=False,
    device='cpu',
):
    """
    Nonnegative Robust PCA via the Stable PCP objective (Zhou et al. 2010):
    min alpha*||L||_* + beta*lambda*||S||_1 + (mu/2)*||M - L - S||_F^2,
    by alternating minimization: the S-update is exact (soft-thresholding);
    the L-update is exact (SVT) followed by a nonnegativity clamp, which
    makes the scheme a heuristic. mu = 1/(sigma*sqrt(2n)) from the estimated
    noise level. Stops on relative change in L.


    Returns:
        L (torch.Tensor): Low-rank nonnegative component
        S (torch.Tensor): Sparse component
        iter_num (int): iterations executed
    """


    # ---- helpers ----
   
    def noise_std(X):
        """
        Input: n x n matrix X
        Output: scalar sigma
        Time Complexity: O(n^3)


        First Step Citation:
        Matan Gavish and David L. Donoho (2014), The Optimal Hard Threshold for
        Singular Values is 4/sqrt(3), IEEE Trans. Information Theory 60(8)


        Second Step Citation:
        Sato and Ono (2024), Joint Background–Anomaly–Noise Decomposition for Robust Hyperspectral Anomaly Detection via Constrained Convex Optimization
        Section IV-I        
       
        """
        n = X.shape[0]


        #First step
        U, s, Vh = torch.linalg.svd(X, full_matrices = False)


        y_med = torch.median(s)
        tau = 2.858 * y_med


        k = (s > tau).sum().item()


        X_tau = (U[:, :k] * s[:k]) @ Vh[:k, :]
        resid = X - X_tau


        #Second step


        triu_idx = torch.triu_indices(X.shape[0], X.shape[1], offset = 1)
        resid_triu = resid[triu_idx[0], triu_idx[1]]
        med = torch.median(resid_triu)
        MAD = torch.median(torch.abs(resid_triu - med)).item()


        return MAD / 0.6745


    def soft_threshold(X, tau):
        """
        elementwise soft-thresholding: sign(x)*max(|x|-tau,0)
        """
        return torch.sign(X) * torch.clamp(X.abs() - tau, min=0.0)


    def svt(X, tau):
        """
        SVT using SVD
        torch.linalg.svd returns U (m,k), S (k,), Vh (k,n)
        """
        U, S, Vh = torch.linalg.svd(X, full_matrices = False)
        S_thresh = torch.clamp(S - tau, min=0.0)
        k_eff = (S_thresh > 0).sum().item()
        if k_eff == 0:
            return torch.zeros_like(X), 0
        # reconstruct with only positive singular values
        U_k = U[:, :k_eff]
        S_k = S_thresh[:k_eff]
        V_k = Vh[:k_eff, :]
        # (U_k * S_k) @ V_k  — use broadcasting for diag
        return (U_k * S_k.unsqueeze(0)) @ V_k, k_eff


    # ---- prep ----


    if isinstance(M, np.ndarray):
        M = torch.tensor(M, dtype=torch.float64)
    else:
        M = M.float()
    M = M.to(device, non_blocking=True)


    assert M.shape[0] == M.shape[1]
    n = M.shape[0]


    if lambda_param is None:
        lambda_param = 1.0 / torch.sqrt(torch.tensor(float(n), device=device))
    if mu is None:
        sigma = noise_std(M)
        sigma = max(sigma, 1e-12)
        mu = 1 / ( np.sqrt(2 * n) * sigma )


    with torch.no_grad():


        L = torch.zeros_like(M)
        S = torch.zeros_like(M)


        iter_num = 0
        for i in range(max_iter):
            iter_num += 1
            # ---- L update via SVT with SVD ----
            X = M - S


            denom = L.pow(2).sum().clamp_min(1e-12)


            L_new, k_eff = svt(X, tau=alpha / mu)
            L_new.clamp_(min=0.0)


            # ---- S update via soft-threshold ----
            S_new = soft_threshold(M - L_new, beta * lambda_param / mu)
           
            # ---- convergence check (every few iters) ----
            if (i % check_every == 0) or (i == max_iter - 1):
                R = L_new - L
                err = (R.pow(2).sum() / denom).sqrt().item()
                if verbose:
                    print(f"Iter {i:4d} | k_eff={k_eff:3d} | err={err:.3e}")
                if err < tol:
                    if verbose:
                        print(f"Converged at iteration {i}.")
                    L, S = L_new, S_new
                    break
            L, S = L_new, S_new


    return L.detach(), S.detach(), iter_num




def find_optimal_components_parafac(tensor, max_components=8):
    """
    Find optimal number of components for CPD using elbow method and KneeLocator.


    Args:
        tensor: Input tensor for CPD decomposition
        max_components: Maximum number of components to test (also the R used
            in the H1 false-positive bound R*p^s -- change both together)


    Returns:
        int: Optimal number of components
    """
    n_components_range = range(2, max_components + 1)
    reconstruction_errors = []


    for n_components in n_components_range:
        print(f'Testing {n_components} components...')
        try:
            weights, factors =parafac(tensor, n_components, normalize_factors=True)
            T_hat = cp_to_tensor((weights, factors))


            tensor_np = tl.to_numpy(tensor)
            T_hat_np = tl.to_numpy(T_hat)
            error = np.linalg.norm(tensor_np - T_hat_np)


            reconstruction_errors.append(error)
        except:
            print(f"CPD failed for {n_components} components.")
            reconstruction_errors.append(np.inf)


    knee_locator = KneeLocator(
        list(n_components_range),
        reconstruction_errors,
        curve='convex',
        direction='decreasing'
    )


    optimal_components = knee_locator.elbow


    if optimal_components is None:
        print("WARNING: No elbow found. Falling back to 2 components.")
        optimal_components = 2


    print(f"Optimal number of components: {optimal_components}")
    return optimal_components






def plotMatrixRaw(matrix, name, originalName, colormap = "viridis", bare = False):
    """
    Plot matrix without any thresholding - shows raw values
    """
    os.makedirs(f"matrix_visualizations/{originalName}", exist_ok=True)


    fig = plt.figure(figsize=(12,10))


    if matrix.shape[1] == 2:
        gs = GridSpec(10, 6, figure=fig)
        ax2 = fig.add_subplot(gs[1:10, 1:4])
    elif matrix.shape[0] == 2:  
        gs = GridSpec(6, 12, figure=fig)  
        ax2 = fig.add_subplot(gs[1:4, 2:10])  
    else:
        gs = GridSpec(10, 12, figure=fig)
        ax2 = fig.add_subplot(gs[1:10, 2:10])


    if matrix.shape[0] <= 10 or matrix.shape[1] <= 10:
        im = ax2.imshow(matrix, cmap=colormap, aspect="auto",
                       extent=[-0.5, matrix.shape[1]-0.5, matrix.shape[0]-0.5, -0.5],
                       interpolation='nearest')


        ax2.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
        ax2.grid(which='minor', color='white', linestyle='-', linewidth=1)


    else:
        im = ax2.imshow(matrix, cmap=colormap, aspect="auto")


    if not bare:
        plt.colorbar(im, ax=ax2, label="Matrix Values")


    if bare:
        # 1-based indexing; "long" axes show only min (1) and max (N)
        n_cols = matrix.shape[1]
        n_rows = matrix.shape[0]
        short_thresh = 20  # axes with <= this many ticks get every label


        if n_cols <= short_thresh:
            ax2.set_xticks(range(n_cols))
            ax2.set_xticklabels([str(j + 1) for j in range(n_cols)])
        else:
            ax2.set_xticks([0, n_cols - 1])
            ax2.set_xticklabels(['1', str(n_cols)])


        if n_rows <= short_thresh:
            ax2.set_yticks(range(n_rows))
            ax2.set_yticklabels([str(i + 1) for i in range(n_rows)])
        else:
            ax2.set_yticks([0, n_rows - 1])
            ax2.set_yticklabels(['1', str(n_rows)])
    else:
        if matrix.shape[1] <= 10:
            ax2.set_xticks(range(matrix.shape[1]))
        else:
            step = max(1, matrix.shape[1] // 10)
            ticks = list(range(0, matrix.shape[1], step))
            if ticks[-1] != matrix.shape[1] - 1:
                ticks.append(matrix.shape[1] - 1)
            ax2.set_xticks(ticks)


        if matrix.shape[0] <= 10:
            ax2.set_yticks(range(matrix.shape[0]))
        else:
            step = max(1, matrix.shape[0] // 10)
            ticks = list(range(0, matrix.shape[0], step))
            if ticks[-1] != matrix.shape[0] - 1:
                ticks.append(matrix.shape[0] - 1)
            ax2.set_yticks(ticks)


    if matrix.shape[1] > 50:
        ax2.tick_params(axis='x', rotation=45)


    matrix_min = np.nanmin(matrix)
    matrix_max = np.nanmax(matrix)
    matrix_mean = np.nanmean(matrix)


    if not bare:
        ax2.set_title(f"Raw Matrix: {name} ({matrix.shape[0]}×{matrix.shape[1]})\n" +
                      f"Range: [{matrix_min:.3f}, {matrix_max:.3f}], Mean: {matrix_mean:.3f}",
                      fontsize=14, pad=40)


    plt.subplots_adjust(top=0.8, bottom=0.25)
    plt.savefig(f"matrix_visualizations/{originalName}/matrix_{name}.svg", dpi=150, bbox_inches='tight')
    plt.close()






def plotMplot(subsequence_length, timeSeriesA, similarityMatrix, threshold, nameA):
       


        histmaximum = np.nanmax(similarityMatrix)


        mplot = similarityMatrix > threshold*histmaximum


        fig = plt.figure(figsize=(12,10))
        gs = GridSpec(10,12, figure=fig)


        ax1 = fig.add_subplot(gs[0:1, 2:10])
        ax1.plot(timeSeriesA)


        ax1.set_xticks([])
        ax1.set_yticks([])


        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.set_facecolor('none')


        ax2 = fig.add_subplot(gs[1:10,2:10])
        im = ax2.imshow(mplot, cmap = "gray", aspect= "auto")
        plt.colorbar(im, ax=ax2)


        ax3 = fig.add_subplot(gs[1:10, 0:1])
        ax3.plot(timeSeriesA, range(len(timeSeriesA)))
       


        ax3.set_xticks([])
        ax3.set_yticks([])


        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.spines["bottom"].set_visible(False)
        ax3.spines["left"].set_visible(False)
        ax3.set_facecolor("none")


        ax3.invert_xaxis()
        ax3.invert_yaxis()


       
        ax2.set_title("Mplot (self-join similarity matrix) for " + nameA + ", with threshold " + str(threshold), fontsize=16, pad=40)
        fig.text(0.5, -0.15, "Subsequence length: " + str(subsequence_length) + ", Time Series length: " + str(np.size(timeSeriesA)), fontsize=14, ha='center', transform=ax2.transAxes)
           
        plt.subplots_adjust(top=0.8, bottom=0.25)


        plt.show()




def processAll(listOfStations, allstationdata, subsequenceLength, Mheight = 258, Mwidth = 258, name = "Dataset"):
   
    from collections import namedtuple
   
    Result = namedtuple("Result", ["low_rank_tensor", "low_rank_factors"])


    LowRankMplots = []
   
    print("===TENSOR CALCULATION===")


    print("calculating mplots...")
    index = 0
    for stationName in listOfStations:
        print(index, stationName)


        series = np.array(allstationdata[stationName].values, dtype=np.float64, order='C', copy=True).reshape(-1,)
        series = np.copy(series, order='C')


        print("beginning RPCA...")


        try:
            mplot = pyscamp.abjoin_matrix(
                np.copy(series),
                np.copy(series),
                subsequenceLength,
                mheight=Mheight,
                mwidth=Mwidth,
                threshold=-1
            )
        except ValueError as e:
            print(f"Error processing {stationName}: {e}")
            continue


        if (np.isnan(mplot).any()):
            print(f"NANs in Mplot: {stationName}")
     
        max_iter = 5000
        L, S, num_iter = nnrobustpca_stable_pcp(mplot, max_iter = max_iter)
        print(f"RPCA done! Converged in {num_iter} iteration(s).")
   




        print(L.norm())
        print(S.norm())


        # Per-slice mean-centering: a slice's mean is its baseline
        # self-similarity level, which carries no information about when
        # patterns recur or which series share them. Left in, its Frobenius
        # energy makes CPD spend a component ranking sensors by overall
        # self-similarity instead of co-clustering structure.
        L_np = L.detach().cpu().numpy()
        L_np = L_np - np.mean(L_np)


        LowRankMplots.append(L_np.astype(np.float32))


        index +=1


    print("calculating Low Rank Tensor...")
    LowRankTensorTemp = np.stack(LowRankMplots, axis = 0).astype(np.float32)
    LowRankTensor = torch.from_numpy(LowRankTensorTemp).to(dtype=torch.float32, device=DEVICE)


    print("Low Rank Tensor Shape:", LowRankTensor.shape)
    LowRankTensor=LowRankTensor/tl.norm(LowRankTensor)




    print("===FACTORIZING LOW RANK TENSOR===")


    print("calculating optimal components...")
    R = find_optimal_components_parafac(LowRankTensor)


    print("calculating factors...")
    LRWeights, LowRankFactors = parafac(LowRankTensor, rank=R, normalize_factors=True)
   
    print("Done!")


    count = 1
    for i in range(len(LowRankFactors)):
        LowRankFactors[i] = LowRankFactors[i].detach().numpy()
        factor = LowRankFactors[i]
        plotMatrixRaw(factor, name + " (Factor Matrix #" + str(count) + ")", name + " (Low Rank)")
        count +=1


   
    return Result(low_rank_tensor = LowRankTensor, low_rank_factors = LowRankFactors)
