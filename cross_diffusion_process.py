import numpy as np

def similarity_normalization(W):
    W_i = np.eye(W.shape[1]) * 0.5
    N = W.shape[1]
    for row in range(N):
        for col in range(row + 1, N):
            W_i[row, col] = W[row, col] / (2 * np.sum(W[row, np.setdiff1d(np.arange(N), [row])]))
    
    W_i = W_i + W_i.T - np.diag(np.diag(W_i))
    return W_i

def sparse_graph(W_i, k):
    N = W_i.shape[0]
    S = np.zeros_like(W_i)
    indies = np.zeros_like(W_i)
    for row in range(N):
        row_sorted = np.argsort(-W_i[row,:])
        for i in range(N):
            for j in row_sorted[1:1+k]:
                S[row, j] = W_i[row, j] / np.sum(W_i[row, row_sorted[1:1+k]])
    
    S = S + S.T - np.diag(np.diag(S))
    return S

def cross_diffusion_process(K_ten, kNN, iter):
    m = K_ten.shape[2]
    K_norm_ten = np.zeros_like(K_ten)
    
    for i in range(m):
        K_norm_ten[:,:,i] = similarity_normalization(K_ten[:,:,i])

    S_ten = np.zeros_like(K_ten)
    for i in range(m):
        S_ten[:,:,i] = sparse_graph(K_norm_ten[:,:,i], kNN)
    
    for t in range(iter):
        for i in range(m):
            S_i = S_ten[:,:,i]
            K_norm_ten[:,:,i] = np.dot(S_i, np.dot((1/(m-1) * np.sum(K_norm_ten[:,:,np.setdiff1d(np.arange(m), [i])], axis=2)), S_i.T))
    
    return K_norm_ten
