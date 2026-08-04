#ifndef IFDFT_IFXC_H
#define IFDFT_IFXC_H

#include <stddef.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(IFDFT_IFXC_BUILD_DLL)
#    define IFDFT_IFXC_API __declspec(dllexport)
#  else
#    define IFDFT_IFXC_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define IFDFT_IFXC_API __attribute__((visibility("default")))
#else
#  define IFDFT_IFXC_API
#endif

#define IFDFT_IFXC_OK 0
#define IFDFT_IFXC_E_INVALID_ARGUMENT -1
#define IFDFT_IFXC_E_IFXC_STATUS -3

#define IFDFT_IFXC_WANT_INTEGRAL_FEATURES 0x01u
#define IFDFT_IFXC_WANT_LOCAL_VALUES 0x02u
#define IFDFT_IFXC_WANT_FIRST_DERIVATIVES 0x04u
#define IFDFT_IFXC_WANT_SECOND_DERIVATIVES 0x08u

IFDFT_IFXC_API int IFDFT_ifxc_version(
    int *major,
    int *minor,
    int *patch);

IFDFT_IFXC_API int IFDFT_ifxc_feature_set_count(size_t *count);

IFDFT_IFXC_API int IFDFT_ifxc_feature_set_key(
    size_t set_index,
    const char **key);

IFDFT_IFXC_API int IFDFT_ifxc_feature_count(
    const char *feature_set_key,
    size_t *count);

IFDFT_IFXC_API int IFDFT_ifxc_feature_key(
    const char *feature_set_key,
    size_t feature_index,
    const char **key);

IFDFT_IFXC_API int IFDFT_ifxc_feature_set_max_deriv_order(
    const char *feature_set_key,
    int *max_deriv_order);

IFDFT_IFXC_API int IFDFT_ifxc_feature_set_xctype(
    const char *feature_set_key,
    const char **xctype);

IFDFT_IFXC_API int IFDFT_ifxc_evaluator_create(
    const char *feature_set_key,
    void **out);

IFDFT_IFXC_API int IFDFT_ifxc_evaluator_destroy(void *ptr);

/* Private runtime control used to honor pyscf.lib.num_threads(). */
IFDFT_IFXC_API int IFDFT_ifxc_evaluator_set_num_threads(
    void *ptr,
    int num_threads);

IFDFT_IFXC_API int IFDFT_ifxc_evaluator_eval_block(
    void *ptr,
    int spin,
    size_t npoints,
    const double *rho,
    const double *sigma,
    const double *tau,
    const double *weights,
    unsigned int flags,
    double *features_integral,
    double *local_values,
    double *d_rho,
    double *d_sigma,
    double *d_tau,
    double *dd_rho_rho,
    double *dd_rho_sigma,
    double *dd_rho_tau,
    double *dd_sigma_sigma,
    double *dd_sigma_tau,
    double *dd_tau_tau,
    char *errbuf,
    size_t errbuf_size);

IFDFT_IFXC_API int IFDFT_ifxc_evaluator_eval_weighted_vvars(
    void *ptr,
    int spin,
    size_t npoints,
    const double *rho,
    const double *sigma,
    const double *tau,
    const double *rho_gradient,
    const double *weights,
    const double *feature_gradient,
    double *weighted_vvars,
    char *errbuf,
    size_t errbuf_size);

IFDFT_IFXC_API int IFDFT_ifxc_evaluator_eval_weighted_vvars_from_pyscf_mgga(
    void *ptr,
    int spin,
    size_t npoints,
    const double *rho_a,
    const double *rho_b,
    int ncomponents,
    const double *weights,
    const double *feature_gradient,
    double *weighted_vvars,
    char *errbuf,
    size_t errbuf_size);

IFDFT_IFXC_API int IFDFT_ifxc_evaluator_eval_weighted_vvars_batch_from_pyscf_mgga(
    void *ptr,
    int spin,
    size_t npoints,
    size_t ncoeff,
    const double *rho_a,
    const double *rho_b,
    int ncomponents,
    const double *weights,
    const double *feature_gradients,
    double *weighted_vvars_batch,
    char *errbuf,
    size_t errbuf_size);

IFDFT_IFXC_API int IFDFT_ifxc_evaluator_eval_features_from_pyscf_mgga(
    void *ptr,
    int spin,
    size_t npoints,
    const double *rho_a,
    const double *rho_b,
    int ncomponents,
    const double *weights,
    double *features_integral,
    char *errbuf,
    size_t errbuf_size);

IFDFT_IFXC_API int IFDFT_ifxc_evaluator_eval_feature_directional_from_pyscf_mgga(
    void *ptr,
    int spin,
    size_t npoints,
    const double *rho0_a,
    const double *rho0_b,
    const double *rho1_a,
    const double *rho1_b,
    int ncomponents,
    const double *weights,
    double *dfeatures,
    char *errbuf,
    size_t errbuf_size);

IFDFT_IFXC_API int IFDFT_ifxc_evaluator_eval_weighted_response_from_pyscf_mgga(
    void *ptr,
    int spin,
    size_t npoints,
    const double *rho0_a,
    const double *rho0_b,
    const double *rho1_a,
    const double *rho1_b,
    int ncomponents,
    const double *weights,
    const double *feature_gradient,
    double *weighted_vvars,
    char *errbuf,
    size_t errbuf_size);

IFDFT_IFXC_API int IFDFT_ifxc_evaluator_eval_response_batch_from_pyscf_mgga(
    void *ptr,
    int spin,
    size_t npoints,
    size_t ntrial,
    const double *rho0_a,
    const double *rho0_b,
    const double *rho1_a_batch,
    const double *rho1_b_batch,
    int ncomponents,
    const double *weights,
    const double *feature_gradient,
    double *weighted_kernel_vvars,
    double *dfeatures,
    char *errbuf,
    size_t errbuf_size);

IFDFT_IFXC_API int IFDFT_ifxc_evaluator_eval_feature_directional_derivative(
    void *ptr,
    int spin,
    size_t npoints,
    const double *rho,
    const double *sigma,
    const double *tau,
    const double *drho,
    const double *dsigma,
    const double *dtau,
    const double *weights,
    double *dfeatures,
    char *errbuf,
    size_t errbuf_size);

IFDFT_IFXC_API int IFDFT_ifxc_evaluator_eval_weighted_response_vvars(
    void *ptr,
    int spin,
    size_t npoints,
    const double *rho,
    const double *sigma,
    const double *tau,
    const double *drho,
    const double *dsigma,
    const double *dtau,
    const double *rho_gradient,
    const double *drho_gradient,
    const double *weights,
    const double *feature_gradient,
    double *weighted_vvars,
    char *errbuf,
    size_t errbuf_size);

IFDFT_IFXC_API int IFDFT_ifxc_eval_weighted_vvars_batch_from_pyscf_mgga(
    void *ptr,
    int spin,
    size_t npoints,
    size_t ncoeff,
    const double *rho_a,
    const double *rho_b,
    int ncomponents,
    const double *weights,
    const double *feature_gradients,
    double *weighted_vvars_batch,
    char *errbuf,
    size_t errbuf_size);

IFDFT_IFXC_API int IFDFT_ifxc_eval_response_batch_from_pyscf_mgga(
    void *ptr,
    int spin,
    size_t npoints,
    size_t ntrial,
    const double *rho0_a,
    const double *rho0_b,
    const double *rho1_a_batch,
    const double *rho1_b_batch,
    int ncomponents,
    const double *weights,
    const double *feature_gradient,
    double *weighted_kernel_vvars,
    double *dfeatures,
    char *errbuf,
    size_t errbuf_size);

#endif
