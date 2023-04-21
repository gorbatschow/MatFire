#pragma once

#include <arrayfire.h>
#include <iostream>
#include <matio.h>
#include <string>
#include <vector>

class MatFire {
public:
  MatFire(const std::string &fname) : _fname{fname} {}

  ~MatFire() { close(); }

  bool read(const std::string &var_name, af::array &arr) {
    if (!openForRead()) {
      return false;
    }

    matvar_t *var_info{Mat_VarRead(_mat, var_name.c_str())};
    if (var_info == nullptr) {
      return false;
    }

    if (var_info->rank > 4) {
      return false;
    }

    af::dim4 dims{1, 1, 1, 1};
    for (std::size_t i{}; i != var_info->rank; ++i) {
      dims.dims[i] = var_info->dims[i];
    }

    if (var_info->data_type == MAT_T_INT8) {
      arr = af::constant<char>({}, dims, b8);
      arr.write(static_cast<const char *>(var_info->data), var_info->nbytes);
    } else if (var_info->data_type == MAT_T_UINT8) {
      arr = af::constant<unsigned char>({}, dims, u8);
      arr.write(static_cast<const unsigned char *>(var_info->data),
                var_info->nbytes);
    } else if (var_info->data_type == MAT_T_INT16) {
      arr = af::constant<short>({}, dims, s16);
      arr.write(static_cast<const short *>(var_info->data), var_info->nbytes);
    } else if (var_info->data_type == MAT_T_UINT16) {
      arr = af::constant<unsigned short>({}, dims, u16);
      arr.write(static_cast<const unsigned short *>(var_info->data),
                var_info->nbytes);
    } else if (var_info->data_type == MAT_T_INT32) {
      arr = af::constant<int>({}, dims, s32);
      arr.write(static_cast<const int *>(var_info->data), var_info->nbytes);
    } else if (var_info->data_type == MAT_T_UINT32) {
      arr = af::constant<unsigned int>({}, dims, u32);
      arr.write(static_cast<const unsigned int *>(var_info->data),
                var_info->nbytes);
    } else if (var_info->data_type == MAT_T_SINGLE) {
      if (var_info->isComplex) {
        af::array arr_re = af::constant<float>({}, dims, f32);
        af::array arr_im = af::constant<float>({}, dims, f32);
        auto buf{static_cast<const mat_complex_split_t *>(var_info->data)};
        arr_re.write(static_cast<const float *>(buf->Re), var_info->nbytes);
        arr_im.write(static_cast<const float *>(buf->Im), var_info->nbytes);
        arr = af::complex(arr_re, arr_im);
      } else {
        arr = af::constant<float>({}, dims, f32);
        arr.write(static_cast<float *>(var_info->data), var_info->nbytes);
      }
    } else if (var_info->data_type == MAT_T_DOUBLE) {
      if (var_info->isComplex) {
        af::array arr_re = af::constant<double>({}, dims, f64);
        af::array arr_im = af::constant<double>({}, dims, f64);
        auto buf{static_cast<const mat_complex_split_t *>(var_info->data)};
        arr_re.write(static_cast<const double *>(buf->Re), var_info->nbytes);
        arr_im.write(static_cast<const double *>(buf->Im), var_info->nbytes);
        arr = af::complex(arr_re, arr_im);
      } else {
        arr = af::constant<double>({}, dims, f64);
        arr.write(static_cast<const double *>(var_info->data),
                  var_info->nbytes);
      }
    } else if (var_info->data_type == MAT_T_INT64) {
      arr = af::constant<long long>({}, dims, s64);
      arr.write(static_cast<const long long *>(var_info->data),
                var_info->nbytes);
    } else if (var_info->data_type == MAT_T_UINT64) {
      arr = af::constant<unsigned long long>({}, dims, u64);
      arr.write(static_cast<const unsigned long long *>(var_info->data),
                var_info->nbytes);
    }

    Mat_VarFree(var_info);

    return true;
  }

  bool write(const std::string &var, const af::array &arr) {
    if (!openForWrite()) {
      return false;
    }

    Mat_VarDelete(_mat, var.c_str());
    matvar_t *var_info{createVar(var, arr)};
    if (var_info == nullptr) {
      return false;
    }

    Mat_VarWrite(_mat, var_info, MAT_COMPRESSION_NONE);
    Mat_VarFree(var_info);
    return true;
  }

  static matvar_t *createVar(const std::string &var, const af::array &arr) {
    matvar_t *var_info{};

    std::size_t rank{arr.numdims()};
    std::size_t dims[4]{1, 1, 1, 1};
    for (std::size_t i{}; i != rank; ++i) {
      dims[i] = arr.dims().dims[i];
    }

    if (arr.type() == b8) {
      std::vector<char> buf(arr.elements());
      arr.host(buf.data());
      var_info = Mat_VarCreate(var.c_str(), MAT_C_INT8, MAT_T_INT8, rank, dims,
                               buf.data(), 0);
    } else if (arr.type() == u8) {
      std::vector<unsigned char> buf(arr.elements());
      arr.host(buf.data());
      var_info = Mat_VarCreate(var.c_str(), MAT_C_UINT8, MAT_T_UINT8, rank,
                               dims, buf.data(), 0);
    } else if (arr.type() == s16) {
      std::vector<short> buf(arr.elements());
      arr.host(buf.data());
      var_info = Mat_VarCreate(var.c_str(), MAT_C_INT16, MAT_T_INT16, rank,
                               dims, buf.data(), 0);
    } else if (arr.type() == u16) {
      std::vector<unsigned short> buf(arr.elements());
      arr.host(buf.data());
      var_info = Mat_VarCreate(var.c_str(), MAT_C_UINT16, MAT_T_UINT16, rank,
                               dims, buf.data(), 0);
    } else if (arr.type() == s32) {
      std::vector<int> buf(arr.elements());
      arr.host(buf.data());
      var_info = Mat_VarCreate(var.c_str(), MAT_C_INT32, MAT_T_INT32, rank,
                               dims, buf.data(), 0);
    } else if (arr.type() == u32) {
      std::vector<unsigned int> buf(arr.elements());
      arr.host(buf.data());
      var_info = Mat_VarCreate(var.c_str(), MAT_C_UINT32, MAT_T_UINT32, rank,
                               dims, buf.data(), 0);
    } else if (arr.type() == f32) {
      std::vector<float> buf(arr.elements());
      arr.host(buf.data());
      var_info = Mat_VarCreate(var.c_str(), MAT_C_SINGLE, MAT_T_SINGLE, rank,
                               dims, buf.data(), 0);
    } else if (arr.type() == c32) {
      std::vector<float> buf_re(arr.elements());
      std::vector<float> buf_im(arr.elements());
      af::real(arr).host(buf_re.data());
      af::imag(arr).host(buf_im.data());
      mat_complex_split_t buf{buf_re.data(), buf_im.data()};
      var_info = Mat_VarCreate(var.c_str(), MAT_C_SINGLE, MAT_T_SINGLE, rank,
                               dims, &buf, MAT_F_COMPLEX);
    } else if (arr.type() == f64) {
      std::vector<double> buf(arr.elements());
      arr.host(buf.data());
      var_info = Mat_VarCreate(var.c_str(), MAT_C_DOUBLE, MAT_T_DOUBLE, rank,
                               dims, buf.data(), 0);
    } else if (arr.type() == c64) {
      std::vector<double> buf_re(arr.elements());
      std::vector<double> buf_im(arr.elements());
      af::real(arr).host(buf_re.data());
      af::imag(arr).host(buf_im.data());
      mat_complex_split_t buf{buf_re.data(), buf_im.data()};
      var_info = Mat_VarCreate(var.c_str(), MAT_C_DOUBLE, MAT_T_DOUBLE, rank,
                               dims, &buf, MAT_F_COMPLEX);
    } else if (arr.type() == s64) {
      std::vector<long long> buf(arr.elements());
      arr.host(buf.data());
      var_info = Mat_VarCreate(var.c_str(), MAT_C_INT64, MAT_T_INT64, rank,
                               dims, buf.data(), 0);
    } else if (arr.type() == u64) {
      std::vector<unsigned long long> buf(arr.elements());
      arr.host(buf.data());
      var_info = Mat_VarCreate(var.c_str(), MAT_C_UINT64, MAT_T_UINT64, rank,
                               dims, buf.data(), 0);
    }

    return var_info;
  }

  bool openForRead() {
    if (_mat != nullptr) {
      return true;
    }

    _mat_mode = MAT_ACC_RDONLY;
    _mat = Mat_Open(_fname.c_str(), _mat_mode);
    if (_mat == nullptr) {
      return false;
    }

    _mat_version = Mat_GetVersion(_mat);
    _fname = Mat_GetFilename(_mat);
    _header = Mat_GetHeader(_mat);

    return true;
  }

  bool openForWrite() {
    if (_mat != nullptr && _mat_mode == MAT_ACC_RDWR) {
      return true;
    } else if (_mat != nullptr && _mat_mode == MAT_ACC_RDONLY) {
      Mat_Close(_mat);
    }

    _mat_mode = MAT_ACC_RDWR;
    _mat = Mat_Open(_fname.c_str(), _mat_mode);
    if (_mat == nullptr) {
      _mat = Mat_CreateVer(_fname.c_str(), nullptr, MAT_FT_MAT5);
    }
    if (_mat == nullptr) {
      return false;
    }

    _mat_version = Mat_GetVersion(_mat);
    _fname = Mat_GetFilename(_mat);
    _header = Mat_GetHeader(_mat);

    return true;
  }

  void close() {
    if (_mat != nullptr) {
      Mat_Close(_mat);
      _mat = nullptr;
    }
  }

  bool getVariabels(std::vector<std::string> &var_list) {
    MatFire mf(_fname);
    if (!mf.openForRead()) {
      return false;
    }
    matvar_t *var_info{};
    while (NULL != (var_info = Mat_VarReadNextInfo(mf._mat))) {
      var_list.push_back(var_info->name);
      Mat_VarFree(var_info);
    }
    mf.close();
    return true;
  }

  static void Test() {
    const std::string fname{"MatFireTest.mat"};

    MatFire mf(fname);

    mf.write("single_1119", af::iota({1, 1, 1, 9}, af::dim4(1), f32));
    mf.write("single_9999", af::iota({9, 9, 9, 9}, af::dim4(1), f32));
    mf.write("single_9111", af::iota({9, 1, 1, 1}, af::dim4(1), f32));
    mf.write("csingle_9999",
             af::complex(af::iota({9, 9, 9, 9}, af::dim4(1), f32),
                         af::iota({9, 9, 9, 9}, af::dim4(1), f32)));

    mf.write("double_9999", af::iota({9, 9, 9, 9}, af::dim4(1), f64));
    mf.write("cdouble_9999",
             af::complex(af::iota({9, 9, 9, 9}, af::dim4(1), f64),
                         af::iota({9, 9, 9, 9}, af::dim4(1), f64)));

    mf.write("int64_9999", af::iota({9, 9, 9, 9}, af::dim4(1), s64));
    mf.write("uint64_9999", af::iota({9, 9, 9, 9}, af::dim4(1), u64));
    mf.write("int32_9999", af::iota({9, 9, 9, 9}, af::dim4(1), s32));
    mf.write("uint32_9999", af::iota({9, 9, 9, 9}, af::dim4(1), u32));
    mf.write("int16_9999", af::iota({9, 9, 9, 9}, af::dim4(1), s16));
    mf.write("uint16_9999", af::iota({9, 9, 9, 9}, af::dim4(1), u16));
    mf.write("int8_9999", af::constant<char>({}, {9, 9, 9, 9}, b8));
    mf.write("uint8_9999", af::constant<unsigned char>({}, {9, 9, 9, 9}, u8));
    mf.close();

    std::vector<std::string> var_list;
    mf.getVariabels(var_list);

    af::array tmp_arr;
    for (auto &var : var_list) {
      MatFire(fname).read(var, tmp_arr);
      MatFire(fname).write(var, tmp_arr);
    }
  }

private:
  mat_t *_mat{};
  mat_acc _mat_mode{};
  mat_ft _mat_version{};
  std::string _fname;
  std::string _header;
};
