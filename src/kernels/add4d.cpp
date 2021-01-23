extern "C" void add4d(double* A, double* B, double* out, int A_lin_offset, int B_lin_offset, int out_lin_offset, int* strides_offsets_out, int dim) {
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0 port = A latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem1 port = B latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem2 port = out latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16
#pragma HLS INTERFACE m_axi offset = slave bundle = plmem0 port = strides_offsets_out latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16

#pragma HLS INTERFACE m_axi offset = slave bundle = control port = A_lin_offset
#pragma HLS INTERFACE m_axi offset = slave bundle = control port = B_lin_offset
#pragma HLS INTERFACE m_axi offset = slave bundle = control port = out_lin_offset
#pragma HLS INTERFACE m_axi offset = slave bundle = control port = dim

	int O_ind, A_ind, B_ind;
	double A_val, B_val;

	int A_offset[4];
	int B_offset[4];
	int out_offset[4];

	int A_stride[4];
	int B_stride[4];
	int out_stride[4];

	int out_end_offset[4];
	int out_shape[4];
	
	for (int i = 0; i<dim; i++){
		A_stride[i] = strides_offsets_out[i];
		B_stride[i] = strides_offsets_out[dim + i];
		out_stride[i] = strides_offsets_out[2*dim + i];

		A_offset[i] = strides_offsets_out[3*dim + i];
		B_offset[i] = strides_offsets_out[4*dim + i];
		out_offset[i] = strides_offsets_out[5*dim + i];

		out_shape[i] = strides_offsets_out[6*dim +i];
		out_end_offset[i] = strides_offsets_out[7*dim + i];
	}

	for (int i=(0 + out_offset[0]); i<(out_shape[0] + out_end_offset[0]); i++){
		for (int j=(0 + out_offset[1]); j<(out_shape[1] + out_end_offset[1]); j++){
			for (int k=(0 + out_offset[2]); k<(out_shape[2] + out_end_offset[2]); k++){
				for (int l=(0 + out_offset[3]); l<(out_shape[3] + out_end_offset[3]); l++){
					A_ind = (i + A_offset[0])*A_stride[0] + (j + A_offset[1])*A_stride[1] + (k + A_offset[2])*A_stride[2] + (l + A_offset[3])*A_stride[3] + A_lin_offset;
					B_ind = (i + B_offset[0])*B_stride[0] + (j + B_offset[1])*B_stride[1] + (k + B_offset[2])*B_stride[2] + (l + B_offset[3])*B_stride[3] + B_lin_offset;
					O_ind = i*out_stride[0] + j*out_stride[1] + k*out_stride[2] + l*out_stride[3] + out_lin_offset;
					
					A_val = A[A_ind];
					B_val = B[B_ind];
					out[O_ind] = A_val + B_val;
				}
			}
		}
	}
}