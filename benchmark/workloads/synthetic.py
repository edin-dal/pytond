@pytond()
def hybrid_mv_nf():
    A = m45.merge(m55, right_on='row_no', left_on='row_no', how='inner')
    B = A.drop(columns=['row_no', 'col8', 'col9'])
    C = B.rename(columns={'col0_x': 'c0', 'col1_x': 'c1', 'col2_x': 'c2', 'col3_x': 'c3', 'col4_x': 'c4', 'col5_x': 'c5', 'col6_x': 'c6', 'col7_x': 'c7', 'col0_y': 'c8', 'col1_y': 'c9', 'col2_y': 'c10', 'col3_y': 'c11', 'col4_y': 'c12', 'col5_y': 'c13', 'col6_y': 'c14', 'col7_y': 'c15'})
    D = C.to_numpy()
    E = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    F = np.einsum('ij,j->i', D, E)
    return F


@pytond()
def hybrid_covar_nf():
    A = m45.merge(m55, right_on='row_no', left_on='row_no', how='inner')
    B = A.drop(columns=['row_no', 'col8', 'col9'])
    C = B.rename(columns={'col0_x': 'c0', 'col1_x': 'c1', 'col2_x': 'c2', 'col3_x': 'c3', 'col4_x': 'c4', 'col5_x': 'c5', 'col6_x': 'c6', 'col7_x': 'c7', 'col0_y': 'c8', 'col1_y': 'c9', 'col2_y': 'c10', 'col3_y': 'c11', 'col4_y': 'c12', 'col5_y': 'c13', 'col6_y': 'c14', 'col7_y': 'c15'})
    D = C.to_numpy()
    E = np.einsum('ij,ik->jk', D, D)
    return E


@pytond()
def hybrid_mv_f():
    A = m45.merge(m55, right_on='row_no', left_on='row_no', how='inner')
    B = A.drop(columns=['row_no', 'col8', 'col9'])
    C = B.rename(columns={'col0_x': 'c0', 'col1_x': 'c1', 'col2_x': 'c2', 'col3_x': 'c3', 'col4_x': 'c4', 'col5_x': 'c5', 'col6_x': 'c6', 'col7_x': 'c7', 'col0_y': 'c8', 'col1_y': 'c9', 'col2_y': 'c10', 'col3_y': 'c11', 'col4_y': 'c12', 'col5_y': 'c13', 'col6_y': 'c14', 'col7_y': 'c15'})
    Filtered = C[(C.c0 == C.c8) & (C.c1 == C.c9)] 
    D = Filtered.to_numpy()
    E = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    F = np.einsum('ij,j->i', D, E)
    return F


@pytond()
def hybrid_covar_f():
    A = m45.merge(m55, right_on='row_no', left_on='row_no', how='inner')
    B = A.drop(columns=['row_no', 'col8', 'col9'])
    C = B.rename(columns={'col0_x': 'c0', 'col1_x': 'c1', 'col2_x': 'c2', 'col3_x': 'c3', 'col4_x': 'c4', 'col5_x': 'c5', 'col6_x': 'c6', 'col7_x': 'c7', 'col0_y': 'c8', 'col1_y': 'c9', 'col2_y': 'c10', 'col3_y': 'c11', 'col4_y': 'c12', 'col5_y': 'c13', 'col6_y': 'c14', 'col7_y': 'c15'})
    Filtered = C[(C.c0 == C.c8) & (C.c1 == C.c9)] 
    D = Filtered.to_numpy()
    E = np.einsum('ij,ik->jk', D, D)
    return E