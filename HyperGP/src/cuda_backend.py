import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

def memcopy_2D(dst, dst_pitch, src, src_pitch, width, height, stream=None,
               src_x_offset=None, src_y_offset=None,
               dst_x_offset=None, dst_y_offset=None):
    mem2dcpy = cuda.Memcpy2D()

    if isinstance(src, np.ndarray):
        mem2dcpy.set_src_host(src)
    elif isinstance(src, cuda.DeviceAllocation):
        mem2dcpy.set_src_device(src)
    if isinstance(dst, cuda.DeviceAllocation):
        mem2dcpy.set_dst_device(dst)
    elif isinstance(dst, np.ndarray):
        mem2dcpy.set_dst_host(dst)

    if src_x_offset is not None:
        mem2dcpy.src_x_in_bytes = src_x_offset
    else:
        mem2dcpy.src_x_in_bytes = 0
    if src_y_offset is not None:
        mem2dcpy.src_y = src_y_offset
    else:
        mem2dcpy.src_y = 0
    if dst_x_offset is not None:
        mem2dcpy.dst_x_in_bytes = dst_x_offset
    else:
        mem2dcpy.dst_x_in_bytes = 0
    if dst_y_offset is not None:
        mem2dcpy.dst_y = dst_y_offset
    else:
        mem2dcpy.dst_y = 0
    mem2dcpy.src_pitch = src_pitch
    mem2dcpy.dst_pitch = dst_pitch
    mem2dcpy.width_in_bytes = width
    mem2dcpy.height = height
    if stream is not None:
        mem2dcpy(stream)
    else:
        mem2dcpy(True)


mod = SourceModule("""
__device__ void add_d(int* exec_unit, double** input, double* output, double* constant, int d_perblock){
    if (exec_unit[2] < 0 && exec_unit[3] < 0){
        double output_constant = constant[0] + constant[1];
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            output[i] = output_constant;
        }
    }
    else if (exec_unit[2] < 0 && exec_unit[3] >= 0){
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            output[i] = constant[0] + input[1][i];
        }
    }
    else if (exec_unit[2] >= 0 && exec_unit[3] < 0){
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            output[i] = constant[1] + input[0][i];
        }
    }
    else {
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            output[i] = input[0][i] + input[1][i];
        }
        //if(blockIdx.x == 0 && threadIdx.x == 0){
        //    printf("d_perblock: %d, %f, %f, %f\\n", d_perblock, output[1], input[0][1], input[1][1]);
        //}
    }
}

__device__ void sub_d(int* exec_unit, double** input, double* output, double* constant, int d_perblock){
    if (exec_unit[2] < 0 && exec_unit[3] < 0){
        double output_constant = constant[0] - constant[1];
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            output[i] = output_constant;
        }
    }
    else if (exec_unit[2] < 0 && exec_unit[3] >= 0){
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            output[i] = constant[0] - input[1][i];
        }
    }
    else if (exec_unit[2] >= 0 && exec_unit[3] < 0){
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            output[i] = constant[1] - input[0][i];
        }
    }
    else {
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            output[i] = input[0][i] - input[1][i];
        }
    }
}

__device__ void mul_d(int* exec_unit, double** input, double* output, double* constant, int d_perblock){
    if (exec_unit[2] < 0 && exec_unit[3] < 0){
        double output_constant = constant[0] * constant[1];
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            output[i] = output_constant;
        }
    }
    else if (exec_unit[2] < 0 && exec_unit[3] >= 0){
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            output[i] = constant[0] * input[1][i];
        }
    }
    else if (exec_unit[2] >= 0 && exec_unit[3] < 0){
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            output[i] = constant[1] * input[0][i];
        }
    }
    else {
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            output[i] = input[0][i] * input[1][i];
        }
    }
}

__device__ void div_d(int* exec_unit, double** input, double* output, double* constant, int d_perblock){
    if (exec_unit[2] < 0 && exec_unit[3] < 0){
        double output_constant = constant[0] / constant[1];
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            output[i] = output_constant;
        }
    }
    else if (exec_unit[2] < 0 && exec_unit[3] >= 0){
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            if(input[1][i] == 0){
                output[i] = constant[0];
            }
            else{
                output[i] = constant[0] / input[1][i];
            }
        }
    }
    else if (exec_unit[2] >= 0 && exec_unit[3] < 0){
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            if(input[0][i] == 0){
                output[i] = constant[1];
            }
            else{
                output[i] = constant[1] / input[0][i];
            }
        }
    }
    else {
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            if(input[1][i] == 0){
                output[i] = input[0][i];
            }
            else{
                output[i] = input[0][i] / input[1][i];
            }
        }
    }
}


__device__ void sin_d(int* exec_unit, double** input, double* output, double* constant, int d_perblock){
    if (exec_unit[2] < 0){
        double output_constant = sin(constant[0]);
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            output[i] = output_constant;
        }
    }
    else {
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            output[i] = sin(input[0][i]);
        }
    }
}

__device__ void cos_d(int* exec_unit, double** input, double* output, double* constant, int d_perblock){
    if (exec_unit[2] < 0){
        double output_constant = cos(constant[0]);
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            output[i] = output_constant;
        }
    }
    else {
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            output[i] = cos(input[0][i]);
        }
    }
}
__device__ void log_fabs(int* exec_unit, double** input, double* output, double* constant, int d_perblock){
    if (exec_unit[2] < 0){
        double output_constant = log(fabs(constant[0]));
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            output[i] = output_constant;
        }
    }
    else {
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            output[i] = log(fabs(input[0][i]));
        }
    }
}
__device__ void exp_d(int* exec_unit, double** input, double* output, double* constant, int d_perblock){
    if (exec_unit[2] < 0){
        double output_constant = exp(constant[0]);
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            output[i] = output_constant;
        }
    }
    else {
        for(int i = threadIdx.x % 32; i < d_perblock; i += 32){
            output[i] = exp(input[0][i]);
        }
    }
}

__global__ void test(double* d_set_, size_t d_pitch, int d_offset){
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("%d, %l\\n", d_offset, d_pitch);
        double* d_set = (double*)(d_set_ + d_pitch * d_offset / 8);
        for(int i = 0; i < 10; ++i){
            printf("d_set: %f\\n", d_set[i]);
        }
    }
}

__global__ void execution_GPU(int* exec_units, int exec_unit_len, int exec_len, double* dataset, size_t* d_pitch_offset, int dlen, double* constants){
    size_t d_pitch = d_pitch_offset[0];
    size_t d_offset_whole = d_pitch_offset[1];
    //printf("d_offset_whole: %f\\n", d_offset_whole);
    double* d_set = (double*)(dataset + d_offset_whole * d_pitch / 8);
    int block_num = gridDim.x, warp_num = blockDim.x / 32, wid = threadIdx.x / 32;
    int d_perblock = dlen / block_num;
    int d_offset = d_perblock * blockIdx.x;
    if(blockIdx.x < dlen % block_num){
        d_perblock += 1;
        d_offset += blockIdx.x;
    }
    else{
        d_offset += dlen % block_num;
    }
    int data_init = 0;
    int exec_id;
    
    int* exec_unit;
    while(wid < exec_len){
        exec_id = exec_unit_len * wid;
        exec_unit = exec_units + exec_id;
        int input_size = exec_unit[1];
        double* output = (double*)(d_set + exec_unit[2 + input_size] * d_pitch / 8) + d_offset;
        
        double* input[2], constant[2];
        switch(exec_unit[0]){
        case 0: //'+'
            /* */
            for(int i = 0; i < input_size; ++i){
                if(exec_unit[2 + i] < 0){
                    constant[i] = constants[-exec_unit[2 + i]];
                }
                else{
                    input[i] = (double*)(d_set + exec_unit[2 + i] * d_pitch / 8) + d_offset;
                }
            }
            //if(exec_unit[2 + input_size] == 102){
            //    printf("here.........%f, %d, %d, %d, %d, %d\\n", output[1], exec_unit[0], exec_unit[1], exec_unit[2], exec_unit[3], exec_unit[4]);
            //}
            //if(blockIdx.x == 0 && threadIdx.x == 0){
            //    printf("here!!!!!!!!%f, %d, %d, %d, %d, %d\\n", output[1], exec_unit[0], exec_unit[1], exec_unit[2], exec_unit[3], exec_unit[4]);
            //    //printf("output: %d, %d, %d, %d, %d, %f\\n", exec_unit[2 + input_size], d_offset, wid, exec_len, warp_num, output[1]);
            //}
            add_d(exec_unit, input, output, constant, d_perblock);
            break;
        case 1: //'-'
        
            /* */
            for(int i = 0; i < input_size; ++i){
                if(exec_unit[2 + i] < 0){
                    constant[i] = constants[-exec_unit[2 + i]];
                }
                else{
                    input[i] = (double*)(d_set + exec_unit[2 + i] * d_pitch / 8) + d_offset;
                }
            }
            
            sub_d(exec_unit, input, output, constant, d_perblock);
            break;
        case 2: //'*'
            /* */
            for(int i = 0; i < input_size; ++i){
                if(exec_unit[2 + i] < 0){
                    constant[i] = constants[-exec_unit[2 + i]];
                }
                else{
                    input[i] = (double*)(d_set + exec_unit[2 + i] * d_pitch / 8) + d_offset;
                }
            }
            
            mul_d(exec_unit, input, output, constant, d_perblock);
            break;
        case 3: //'/'
            /* */
            for(int i = 0; i < input_size; ++i){
                if(exec_unit[2 + i] < 0){
                    constant[i] = constants[-exec_unit[2 + i]];
                }
                else{
                    input[i] = (double*)(d_set + exec_unit[2 + i] * d_pitch / 8) + d_offset;
                }
            }
            
            div_d(exec_unit, input, output, constant, d_perblock);
            break;
        case 4: //'sin'
            /* */
            constant[0] = constants[-exec_unit[2]];
            input[0] = (double*)(d_set + exec_unit[2] * d_pitch / 8) + d_offset;
            
            sin_d(exec_unit, input, output, constant, d_perblock);
            break;
        case 5: //'cos'
            /* */
            constant[0] = constants[-exec_unit[2]];
            input[0] = (double*)(d_set + exec_unit[2] * d_pitch / 8) + d_offset;
            
            cos_d(exec_unit, input, output, constant, d_perblock);
            break;
        case 6: //'log'
            /* */
            constant[0] = constants[-exec_unit[2]];
            input[0] = (double*)(d_set + exec_unit[2] * d_pitch / 8) + d_offset;
            
            log_fabs(exec_unit, input, output, constant, d_perblock);
            break;
        case 7: //'exp'
            /* */
            constant[0] = constants[-exec_unit[2]];
            input[0] = (double*)(d_set + exec_unit[2] * d_pitch / 8) + d_offset;
            
            exp_d(exec_unit, input, output, constant, d_perblock);
            break;
        default:
            printf("operator id out of range..\\n");
            break;
        }
        wid += warp_num;
    }
}


""")