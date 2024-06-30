@pytond()
def covar_dense():
    return np.einsum('ij,ik->jk', m_dense, m_dense)

@pytond(input_layouts={'m_coo': 'sparse'})
def covar_coo():
    return np.einsum('ij,ik->jk', m_coo, m_coo)