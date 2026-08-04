#include "ifdft_ifxc.h"

#include "ifxc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define IFDFT_MIN_POINTS_PER_WORKER 256u

typedef struct {
  const char *key;
  const char *xctype;
  int feature_set;
} ifdft_feature_set_entry;

static const ifdft_feature_set_entry IFDFT_FEATURE_SETS[] = {
  {"ml25", "MGGA", IFXC_FEATURE_SET_ML25},
  {"ml26", "MGGA", IFXC_FEATURE_SET_ML26}
};

static const size_t IFDFT_FEATURE_SET_COUNT =
    sizeof(IFDFT_FEATURE_SETS) / sizeof(IFDFT_FEATURE_SETS[0]);

typedef struct {
  double *rho;
  double *sigma;
  double *tau;
  double *rho_gradient;
  double *drho;
  double *dsigma;
  double *dtau;
  double *drho_gradient;
  double *d_rho;
  double *d_sigma;
  double *d_tau;
  double *feature_d_rho;
  double *feature_d_sigma;
  double *feature_d_tau;
  double *dd_rho_rho;
  double *dd_rho_sigma;
  double *dd_rho_tau;
  double *dd_sigma_sigma;
  double *dd_sigma_tau;
  double *dd_tau_tau;
  double *unit_weights;
  double *weighted_vvars;
  double *features;
  size_t rho_capacity;
  size_t sigma_capacity;
  size_t tau_capacity;
  size_t rho_gradient_capacity;
  size_t drho_capacity;
  size_t dsigma_capacity;
  size_t dtau_capacity;
  size_t drho_gradient_capacity;
  size_t d_rho_capacity;
  size_t d_sigma_capacity;
  size_t d_tau_capacity;
  size_t feature_d_rho_capacity;
  size_t feature_d_sigma_capacity;
  size_t feature_d_tau_capacity;
  size_t dd_rho_rho_capacity;
  size_t dd_rho_sigma_capacity;
  size_t dd_rho_tau_capacity;
  size_t dd_sigma_sigma_capacity;
  size_t dd_sigma_tau_capacity;
  size_t dd_tau_tau_capacity;
  size_t unit_weights_capacity;
  size_t weighted_vvars_capacity;
  size_t features_capacity;
} IFDFT_Workspace;

typedef struct {
  ifxc_func_type funcs[2];
  int initialized[2];
  IFDFT_Workspace workspace;
} IFDFT_IFXC_Worker;

typedef struct {
  int feature_set;
  int requested_workers;
  size_t worker_capacity;
  IFDFT_IFXC_Worker *workers;
} IFDFT_IFXC_Evaluator;

static void
ifdft_set_error(char *errbuf, size_t errbuf_size, const char *message)
{
  if(errbuf == NULL || errbuf_size == 0){
    return;
  }
  if(message == NULL){
    message = "unknown error";
  }
  snprintf(errbuf, errbuf_size, "%s", message);
}

static int
ifdft_status(ifxc_status status, char *errbuf, size_t errbuf_size);

static int
ifdft_workspace_require(
    double **buffer,
    size_t *capacity,
    size_t needed,
    const char *label,
    char *errbuf,
    size_t errbuf_size);

static void
ifdft_workspace_free(IFDFT_Workspace *workspace);

static int
ifdft_resolve_feature_set_key(
    const char *key,
    const ifdft_feature_set_entry **entry);

static ifxc_nspin
ifdft_nspin_from_spin(int spin);

static int
ifdft_evaluator_get_func(
    IFDFT_IFXC_Evaluator *evaluator,
    int spin,
    ifxc_func_type **func,
    char *errbuf,
    size_t errbuf_size);

static int
ifdft_evaluator_get_worker_func(
    IFDFT_IFXC_Evaluator *evaluator,
    size_t worker_index,
    int spin,
    ifxc_func_type **func,
    char *errbuf,
    size_t errbuf_size);

static int
ifdft_evaluator_require_workers(
    IFDFT_IFXC_Evaluator *evaluator,
    size_t count,
    char *errbuf,
    size_t errbuf_size);

static int
ifdft_eval_block_with_func(
    const ifxc_func_type *func,
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

static int
ifdft_eval_weighted_vvars_with_func(
    const ifxc_func_type *func,
    IFDFT_Workspace *workspace,
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

static int
ifdft_eval_feature_directional_derivative_with_func(
    const ifxc_func_type *func,
    IFDFT_Workspace *workspace,
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

static int
ifdft_eval_weighted_vvars_from_pyscf_mgga_with_func(
    const ifxc_func_type *func,
    IFDFT_Workspace *workspace,
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

static int
ifdft_eval_weighted_vvars_batch_from_pyscf_mgga_with_func(
    const ifxc_func_type *func,
    IFDFT_Workspace *workspace,
    int spin,
    size_t npoints,
    size_t source_npoints,
    size_t source_start,
    size_t ncoeff,
    const double *rho_a,
    const double *rho_b,
    int ncomponents,
    const double *weights,
    const double *feature_gradients,
    double *weighted_vvars_batch,
    char *errbuf,
    size_t errbuf_size);

static int
ifdft_eval_features_from_pyscf_mgga_with_func(
    const ifxc_func_type *func,
    IFDFT_Workspace *workspace,
    int spin,
    size_t npoints,
    const double *rho_a,
    const double *rho_b,
    int ncomponents,
    const double *weights,
    double *features_integral,
    char *errbuf,
    size_t errbuf_size);

static int
ifdft_parallel_eval_weighted_vvars_from_pyscf_mgga(
    IFDFT_IFXC_Evaluator *evaluator,
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

static int
ifdft_spin_index(int spin);

static int
ifdft_validate_pyscf_mgga_input(
    int spin,
    size_t npoints,
    const double *rho_a,
    const double *rho_b,
    int ncomponents,
    char *errbuf,
    size_t errbuf_size);

static size_t
ifdft_parallel_worker_count(
    const IFDFT_IFXC_Evaluator *evaluator,
    size_t npoints);

static int
ifdft_parallel_eval_weighted_vvars_batch_from_pyscf_mgga(
    IFDFT_IFXC_Evaluator *evaluator,
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
    size_t errbuf_size)
{
  size_t nworkers;
  size_t output_components;
  size_t worker_index;
  int *statuses = NULL;
  int rc;

  if(evaluator == NULL || ncoeff == 0 || weights == NULL
      || feature_gradients == NULL || weighted_vvars_batch == NULL){
    ifdft_set_error(errbuf, errbuf_size,
                    "missing parallel PySCF MGGA weighted-vvar batch input");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  rc = ifdft_validate_pyscf_mgga_input(
      spin, npoints, rho_a, rho_b, ncomponents, errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  nworkers = ifdft_parallel_worker_count(evaluator, npoints);
  rc = ifdft_evaluator_require_workers(evaluator, nworkers, errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  {
    ifxc_func_type *func = NULL;
    rc = ifdft_evaluator_get_worker_func(
        evaluator, 0, spin, &func, errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      return rc;
    }
    if(nworkers == 1){
      return ifdft_eval_weighted_vvars_batch_from_pyscf_mgga_with_func(
          func, &evaluator->workers[0].workspace, spin,
          npoints, npoints, 0, ncoeff, rho_a, rho_b, ncomponents,
          weights, feature_gradients, weighted_vvars_batch, errbuf, errbuf_size);
    }
  }

  output_components = (spin == 0) ? 5 : 10;
  statuses = (int *)calloc(nworkers, sizeof(int));
  if(statuses == NULL){
    ifdft_set_error(errbuf, errbuf_size, "failed to allocate worker statuses");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  for(worker_index = 0; worker_index < nworkers; ++worker_index){
    ifxc_func_type *func = NULL;
    size_t start = npoints * worker_index / nworkers;
    size_t end = npoints * (worker_index + 1) / nworkers;
    size_t chunk_size = end - start;
    rc = ifdft_evaluator_get_worker_func(
        evaluator, worker_index, spin, &func, errbuf, errbuf_size);
    if(rc == IFDFT_IFXC_OK){
      rc = ifdft_workspace_require(
          &evaluator->workers[worker_index].workspace.weighted_vvars,
          &evaluator->workers[worker_index].workspace.weighted_vvars_capacity,
          ncoeff * output_components * chunk_size,
          "worker weighted-vvar batch", errbuf, errbuf_size);
    }
    if(rc != IFDFT_IFXC_OK){
      free(statuses);
      return rc;
    }
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1) num_threads((int)nworkers)
#endif
  for(worker_index = 0; worker_index < nworkers; ++worker_index){
    IFDFT_IFXC_Worker *worker = &evaluator->workers[worker_index];
    size_t start = npoints * worker_index / nworkers;
    size_t end = npoints * (worker_index + 1) / nworkers;
    size_t chunk_size = end - start;
    statuses[worker_index] =
        ifdft_eval_weighted_vvars_batch_from_pyscf_mgga_with_func(
            &worker->funcs[ifdft_spin_index(spin)], &worker->workspace, spin,
            chunk_size, npoints, start, ncoeff, rho_a, rho_b, ncomponents,
            weights + start, feature_gradients, worker->workspace.weighted_vvars,
            NULL, 0);
  }

  rc = IFDFT_IFXC_OK;
  for(worker_index = 0; worker_index < nworkers; ++worker_index){
    size_t start = npoints * worker_index / nworkers;
    size_t end = npoints * (worker_index + 1) / nworkers;
    size_t chunk_size = end - start;
    size_t coeff_index;
    if(statuses[worker_index] != IFDFT_IFXC_OK){
      rc = statuses[worker_index];
      ifdft_set_error(errbuf, errbuf_size,
                      "parallel PySCF MGGA weighted-vvar batch worker failed");
      break;
    }
    for(coeff_index = 0; coeff_index < ncoeff; ++coeff_index){
      size_t component;
      for(component = 0; component < output_components; ++component){
        memcpy(
            weighted_vvars_batch
                + (coeff_index * output_components + component) * npoints + start,
            evaluator->workers[worker_index].workspace.weighted_vvars
                + (coeff_index * output_components + component) * chunk_size,
            chunk_size * sizeof(double));
      }
    }
  }
  free(statuses);
  return rc;
}

static int
ifdft_parallel_eval_features_from_pyscf_mgga(
    IFDFT_IFXC_Evaluator *evaluator,
    int spin,
    size_t npoints,
    const double *rho_a,
    const double *rho_b,
    int ncomponents,
    const double *weights,
    double *features_integral,
    char *errbuf,
    size_t errbuf_size);

static int
ifdft_eval_feature_directional_from_pyscf_mgga_with_func(
    const ifxc_func_type *func,
    IFDFT_Workspace *workspace,
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

static int
ifdft_eval_weighted_response_from_pyscf_mgga_with_func(
    const ifxc_func_type *func,
    IFDFT_Workspace *workspace,
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

static int
ifdft_eval_response_batch_from_pyscf_mgga_with_func(
    const ifxc_func_type *func,
    IFDFT_Workspace *workspace,
    int spin,
    size_t npoints,
    size_t source_npoints,
    size_t source_start,
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

static int
ifdft_parallel_eval_response_batch_from_pyscf_mgga(
    IFDFT_IFXC_Evaluator *evaluator,
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

static int
ifdft_eval_weighted_response_vvars_with_func(
    const ifxc_func_type *func,
    IFDFT_Workspace *workspace,
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

int
IFDFT_ifxc_evaluator_create(const char *feature_set_key, void **out)
{
  const ifdft_feature_set_entry *entry = NULL;
  IFDFT_IFXC_Evaluator *evaluator = NULL;
  int rc;

  if(out == NULL){
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  *out = NULL;
  rc = ifdft_resolve_feature_set_key(feature_set_key, &entry);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }

  evaluator = (IFDFT_IFXC_Evaluator *)calloc(1, sizeof(IFDFT_IFXC_Evaluator));
  if(evaluator == NULL){
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  evaluator->feature_set = entry->feature_set;
  evaluator->requested_workers = 1;
  evaluator->workers = (IFDFT_IFXC_Worker *)calloc(1, sizeof(IFDFT_IFXC_Worker));
  if(evaluator->workers == NULL){
    free(evaluator);
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  evaluator->worker_capacity = 1;
  *out = evaluator;
  return IFDFT_IFXC_OK;
}

int
IFDFT_ifxc_evaluator_destroy(void *ptr)
{
  IFDFT_IFXC_Evaluator *evaluator = (IFDFT_IFXC_Evaluator *)ptr;
  size_t worker_index;

  if(evaluator == NULL){
    return IFDFT_IFXC_OK;
  }
  for(worker_index = 0; worker_index < evaluator->worker_capacity; ++worker_index){
    size_t spin_index;
    IFDFT_IFXC_Worker *worker = &evaluator->workers[worker_index];
    for(spin_index = 0; spin_index < 2; ++spin_index){
      if(worker->initialized[spin_index]){
        ifxc_end(&worker->funcs[spin_index]);
        worker->initialized[spin_index] = 0;
      }
    }
    ifdft_workspace_free(&worker->workspace);
  }
  free(evaluator->workers);
  free(evaluator);
  return IFDFT_IFXC_OK;
}

int
IFDFT_ifxc_evaluator_set_num_threads(void *ptr, int num_threads)
{
  IFDFT_IFXC_Evaluator *evaluator = (IFDFT_IFXC_Evaluator *)ptr;
  if(evaluator == NULL || num_threads < 1){
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  evaluator->requested_workers = num_threads;
  return IFDFT_IFXC_OK;
}

int
IFDFT_ifxc_evaluator_eval_block(
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
    size_t errbuf_size)
{
  IFDFT_IFXC_Evaluator *evaluator = (IFDFT_IFXC_Evaluator *)ptr;
  ifxc_func_type *func = NULL;
  int rc;

  if(errbuf != NULL && errbuf_size > 0){
    errbuf[0] = '\0';
  }
  rc = ifdft_evaluator_get_func(evaluator, spin, &func, errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  return ifdft_eval_block_with_func(
      func,
      spin,
      npoints,
      rho,
      sigma,
      tau,
      weights,
      flags,
      features_integral,
      local_values,
      d_rho,
      d_sigma,
      d_tau,
      dd_rho_rho,
      dd_rho_sigma,
      dd_rho_tau,
      dd_sigma_sigma,
      dd_sigma_tau,
      dd_tau_tau,
      errbuf,
      errbuf_size);
}

int
IFDFT_ifxc_evaluator_eval_weighted_vvars(
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
    size_t errbuf_size)
{
  IFDFT_IFXC_Evaluator *evaluator = (IFDFT_IFXC_Evaluator *)ptr;
  ifxc_func_type *func = NULL;
  int rc;

  if(errbuf != NULL && errbuf_size > 0){
    errbuf[0] = '\0';
  }
  rc = ifdft_evaluator_get_func(evaluator, spin, &func, errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  return ifdft_eval_weighted_vvars_with_func(
      func,
      &evaluator->workers[0].workspace,
      spin,
      npoints,
      rho,
      sigma,
      tau,
      rho_gradient,
      weights,
      feature_gradient,
      weighted_vvars,
      errbuf,
      errbuf_size);
}

int
IFDFT_ifxc_evaluator_eval_weighted_vvars_from_pyscf_mgga(
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
    size_t errbuf_size)
{
  IFDFT_IFXC_Evaluator *evaluator = (IFDFT_IFXC_Evaluator *)ptr;

  if(errbuf != NULL && errbuf_size > 0){
    errbuf[0] = '\0';
  }
  return ifdft_parallel_eval_weighted_vvars_from_pyscf_mgga(
      evaluator,
      spin,
      npoints,
      rho_a,
      rho_b,
      ncomponents,
      weights,
      feature_gradient,
      weighted_vvars,
      errbuf,
      errbuf_size);
}

int
IFDFT_ifxc_evaluator_eval_weighted_vvars_batch_from_pyscf_mgga(
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
    size_t errbuf_size)
{
  IFDFT_IFXC_Evaluator *evaluator = (IFDFT_IFXC_Evaluator *)ptr;
  if(errbuf != NULL && errbuf_size > 0){
    errbuf[0] = '\0';
  }
  return ifdft_parallel_eval_weighted_vvars_batch_from_pyscf_mgga(
      evaluator,
      spin,
      npoints,
      ncoeff,
      rho_a,
      rho_b,
      ncomponents,
      weights,
      feature_gradients,
      weighted_vvars_batch,
      errbuf,
      errbuf_size);
}

int
IFDFT_ifxc_evaluator_eval_features_from_pyscf_mgga(
    void *ptr,
    int spin,
    size_t npoints,
    const double *rho_a,
    const double *rho_b,
    int ncomponents,
    const double *weights,
    double *features_integral,
    char *errbuf,
    size_t errbuf_size)
{
  IFDFT_IFXC_Evaluator *evaluator = (IFDFT_IFXC_Evaluator *)ptr;

  if(errbuf != NULL && errbuf_size > 0){
    errbuf[0] = '\0';
  }
  return ifdft_parallel_eval_features_from_pyscf_mgga(
      evaluator,
      spin,
      npoints,
      rho_a,
      rho_b,
      ncomponents,
      weights,
      features_integral,
      errbuf,
      errbuf_size);
}

int
IFDFT_ifxc_evaluator_eval_feature_directional_from_pyscf_mgga(
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
    size_t errbuf_size)
{
  IFDFT_IFXC_Evaluator *evaluator = (IFDFT_IFXC_Evaluator *)ptr;
  ifxc_func_type *func = NULL;
  int rc;

  if(errbuf != NULL && errbuf_size > 0){
    errbuf[0] = '\0';
  }
  rc = ifdft_evaluator_get_func(evaluator, spin, &func, errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  return ifdft_eval_feature_directional_from_pyscf_mgga_with_func(
      func,
      &evaluator->workers[0].workspace,
      spin,
      npoints,
      rho0_a,
      rho0_b,
      rho1_a,
      rho1_b,
      ncomponents,
      weights,
      dfeatures,
      errbuf,
      errbuf_size);
}

int
IFDFT_ifxc_evaluator_eval_weighted_response_from_pyscf_mgga(
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
    size_t errbuf_size)
{
  IFDFT_IFXC_Evaluator *evaluator = (IFDFT_IFXC_Evaluator *)ptr;
  ifxc_func_type *func = NULL;
  int rc;

  if(errbuf != NULL && errbuf_size > 0){
    errbuf[0] = '\0';
  }
  rc = ifdft_evaluator_get_func(evaluator, spin, &func, errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  return ifdft_eval_weighted_response_from_pyscf_mgga_with_func(
      func,
      &evaluator->workers[0].workspace,
      spin,
      npoints,
      rho0_a,
      rho0_b,
      rho1_a,
      rho1_b,
      ncomponents,
      weights,
      feature_gradient,
      weighted_vvars,
      errbuf,
      errbuf_size);
}

int
IFDFT_ifxc_evaluator_eval_response_batch_from_pyscf_mgga(
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
    size_t errbuf_size)
{
  IFDFT_IFXC_Evaluator *evaluator = (IFDFT_IFXC_Evaluator *)ptr;
  if(errbuf != NULL && errbuf_size > 0){
    errbuf[0] = '\0';
  }
  return ifdft_parallel_eval_response_batch_from_pyscf_mgga(
      evaluator,
      spin,
      npoints,
      ntrial,
      rho0_a,
      rho0_b,
      rho1_a_batch,
      rho1_b_batch,
      ncomponents,
      weights,
      feature_gradient,
      weighted_kernel_vvars,
      dfeatures,
      errbuf,
      errbuf_size);
}

int
IFDFT_ifxc_evaluator_eval_feature_directional_derivative(
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
    size_t errbuf_size)
{
  IFDFT_IFXC_Evaluator *evaluator = (IFDFT_IFXC_Evaluator *)ptr;
  ifxc_func_type *func = NULL;
  int rc;

  if(errbuf != NULL && errbuf_size > 0){
    errbuf[0] = '\0';
  }
  rc = ifdft_evaluator_get_func(evaluator, spin, &func, errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  return ifdft_eval_feature_directional_derivative_with_func(
      func,
      &evaluator->workers[0].workspace,
      spin,
      npoints,
      rho,
      sigma,
      tau,
      drho,
      dsigma,
      dtau,
      weights,
      dfeatures,
      errbuf,
      errbuf_size);
}

int
IFDFT_ifxc_evaluator_eval_weighted_response_vvars(
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
    size_t errbuf_size)
{
  IFDFT_IFXC_Evaluator *evaluator = (IFDFT_IFXC_Evaluator *)ptr;
  ifxc_func_type *func = NULL;
  int rc;

  if(errbuf != NULL && errbuf_size > 0){
    errbuf[0] = '\0';
  }
  rc = ifdft_evaluator_get_func(evaluator, spin, &func, errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  return ifdft_eval_weighted_response_vvars_with_func(
      func,
      &evaluator->workers[0].workspace,
      spin,
      npoints,
      rho,
      sigma,
      tau,
      drho,
      dsigma,
      dtau,
      rho_gradient,
      drho_gradient,
      weights,
      feature_gradient,
      weighted_vvars,
      errbuf,
      errbuf_size);
}

int
IFDFT_ifxc_eval_weighted_vvars_batch_from_pyscf_mgga(
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
    size_t errbuf_size)
{
  return IFDFT_ifxc_evaluator_eval_weighted_vvars_batch_from_pyscf_mgga(
      ptr,
      spin,
      npoints,
      ncoeff,
      rho_a,
      rho_b,
      ncomponents,
      weights,
      feature_gradients,
      weighted_vvars_batch,
      errbuf,
      errbuf_size);
}

int
IFDFT_ifxc_eval_response_batch_from_pyscf_mgga(
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
    size_t errbuf_size)
{
  return IFDFT_ifxc_evaluator_eval_response_batch_from_pyscf_mgga(
      ptr,
      spin,
      npoints,
      ntrial,
      rho0_a,
      rho0_b,
      rho1_a_batch,
      rho1_b_batch,
      ncomponents,
      weights,
      feature_gradient,
      weighted_kernel_vvars,
      dfeatures,
      errbuf,
      errbuf_size);
}

static int
ifdft_status(ifxc_status status, char *errbuf, size_t errbuf_size)
{
  if(status == IFXC_OK){
    return IFDFT_IFXC_OK;
  }
  ifdft_set_error(errbuf, errbuf_size, ifxc_strerror(status));
  return IFDFT_IFXC_E_IFXC_STATUS;
}

static int
ifdft_workspace_require(
    double **buffer,
    size_t *capacity,
    size_t needed,
    const char *label,
    char *errbuf,
    size_t errbuf_size)
{
  double *new_buffer;

  if(buffer == NULL || capacity == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing workspace buffer pointer");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  if(needed <= *capacity){
    return IFDFT_IFXC_OK;
  }
  if(needed > ((size_t)-1) / sizeof(double)){
    ifdft_set_error(errbuf, errbuf_size, "workspace buffer size overflow");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  new_buffer = (double *)realloc(*buffer, needed * sizeof(double));
  if(new_buffer == NULL){
    char message[160];
    snprintf(message, sizeof(message), "failed to grow %s workspace buffer", label);
    ifdft_set_error(errbuf, errbuf_size, message);
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  *buffer = new_buffer;
  *capacity = needed;
  return IFDFT_IFXC_OK;
}

static void
ifdft_workspace_free(IFDFT_Workspace *workspace)
{
  if(workspace == NULL){
    return;
  }
  free(workspace->rho);
  free(workspace->sigma);
  free(workspace->tau);
  free(workspace->rho_gradient);
  free(workspace->drho);
  free(workspace->dsigma);
  free(workspace->dtau);
  free(workspace->drho_gradient);
  free(workspace->d_rho);
  free(workspace->d_sigma);
  free(workspace->d_tau);
  free(workspace->feature_d_rho);
  free(workspace->feature_d_sigma);
  free(workspace->feature_d_tau);
  free(workspace->dd_rho_rho);
  free(workspace->dd_rho_sigma);
  free(workspace->dd_rho_tau);
  free(workspace->dd_sigma_sigma);
  free(workspace->dd_sigma_tau);
  free(workspace->dd_tau_tau);
  free(workspace->unit_weights);
  free(workspace->weighted_vvars);
  free(workspace->features);
  memset(workspace, 0, sizeof(*workspace));
}

static int
ifdft_resolve_feature_set_key(
    const char *key,
    const ifdft_feature_set_entry **entry)
{
  const char *canonical = key;
  size_t idx;

  if(entry == NULL || key == NULL){
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  *entry = NULL;

  if(strcmp(key, "mgga_xc_ml25") == 0 || strcmp(key, "MGGA_XC_ML25") == 0){
    canonical = "ml25";
  }else if(strcmp(key, "mgga_xc_ml26") == 0 || strcmp(key, "MGGA_XC_ML26") == 0){
    canonical = "ml26";
  }

  for(idx = 0; idx < IFDFT_FEATURE_SET_COUNT; ++idx){
    if(strcmp(IFDFT_FEATURE_SETS[idx].key, canonical) == 0){
      *entry = &IFDFT_FEATURE_SETS[idx];
      return IFDFT_IFXC_OK;
    }
  }
  return IFDFT_IFXC_E_IFXC_STATUS;
}

static ifxc_nspin
ifdft_nspin_from_spin(int spin)
{
  if(spin == 0){
    return IFXC_UNPOLARIZED;
  }
  if(spin == 1){
    return IFXC_POLARIZED;
  }
  return 0;
}

static int
ifdft_spin_index(int spin)
{
  if(spin == 0){
    return 0;
  }
  if(spin == 1){
    return 1;
  }
  return -1;
}

static size_t
ifdft_sym_count(size_t ncomp)
{
  return ncomp * (ncomp + 1) / 2;
}

static size_t
ifdft_feature_deriv_index(
    size_t point,
    size_t feature,
    size_t nfeatures,
    size_t ncomponents,
    size_t component)
{
  return (point * nfeatures + feature) * ncomponents + component;
}

static int
ifdft_evaluator_get_func(
    IFDFT_IFXC_Evaluator *evaluator,
    int spin,
    ifxc_func_type **func,
    char *errbuf,
    size_t errbuf_size)
{
  return ifdft_evaluator_get_worker_func(
      evaluator, 0, spin, func, errbuf, errbuf_size);
}

static int
ifdft_evaluator_require_workers(
    IFDFT_IFXC_Evaluator *evaluator,
    size_t count,
    char *errbuf,
    size_t errbuf_size)
{
  IFDFT_IFXC_Worker *workers;
  size_t old_capacity;

  if(evaluator == NULL || count == 0){
    ifdft_set_error(errbuf, errbuf_size, "invalid evaluator worker request");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  if(count <= evaluator->worker_capacity){
    return IFDFT_IFXC_OK;
  }
  old_capacity = evaluator->worker_capacity;
  workers = (IFDFT_IFXC_Worker *)realloc(
      evaluator->workers, count * sizeof(IFDFT_IFXC_Worker));
  if(workers == NULL){
    ifdft_set_error(errbuf, errbuf_size, "failed to allocate evaluator workers");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  memset(workers + old_capacity, 0,
         (count - old_capacity) * sizeof(IFDFT_IFXC_Worker));
  evaluator->workers = workers;
  evaluator->worker_capacity = count;
  return IFDFT_IFXC_OK;
}

static int
ifdft_evaluator_get_worker_func(
    IFDFT_IFXC_Evaluator *evaluator,
    size_t worker_index,
    int spin,
    ifxc_func_type **func,
    char *errbuf,
    size_t errbuf_size)
{
  int idx = ifdft_spin_index(spin);
  ifxc_nspin nspin;
  ifxc_status status;
  IFDFT_IFXC_Worker *worker;

  if(evaluator == NULL || func == NULL || idx < 0 ||
     worker_index >= evaluator->worker_capacity){
    ifdft_set_error(errbuf, errbuf_size, "invalid evaluator or spin");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  worker = &evaluator->workers[worker_index];

  if(!worker->initialized[idx]){
    nspin = ifdft_nspin_from_spin(spin);
    if(nspin == 0){
      ifdft_set_error(errbuf, errbuf_size, "spin must be 0 for RKS or 1 for UKS");
      return IFDFT_IFXC_E_INVALID_ARGUMENT;
    }
    status = ifxc_init(&worker->funcs[idx], evaluator->feature_set, nspin);
    if(status != IFXC_OK){
      return ifdft_status(status, errbuf, errbuf_size);
    }
    worker->initialized[idx] = 1;
  }

  *func = &worker->funcs[idx];
  return IFDFT_IFXC_OK;
}

static int
ifdft_validate_directional_args(
    const ifxc_dimensions_t *dims,
    int spin,
    size_t npoints,
    const double *rho,
    const double *sigma,
    const double *tau,
    const double *drho,
    const double *dsigma,
    const double *dtau,
    const double *weights,
    const double *dfeatures,
    char *errbuf,
    size_t errbuf_size)
{
  if(dims == NULL || npoints == 0 || rho == NULL || sigma == NULL || tau == NULL
      || drho == NULL || dsigma == NULL || dtau == NULL || weights == NULL
      || dfeatures == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing feature directional derivative input or output buffer");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  if(spin == 0){
    if(dims->rho != 1 || dims->sigma != 1 || dims->tau != 1){
      ifdft_set_error(errbuf, errbuf_size, "RKS feature derivative requires unpolarized MGGA dimensions");
      return IFDFT_IFXC_E_INVALID_ARGUMENT;
    }
    return IFDFT_IFXC_OK;
  }
  if(spin == 1){
    if(dims->rho != 2 || dims->sigma != 3 || dims->tau != 2){
      ifdft_set_error(errbuf, errbuf_size, "UKS feature derivative requires polarized MGGA dimensions");
      return IFDFT_IFXC_E_INVALID_ARGUMENT;
    }
    return IFDFT_IFXC_OK;
  }
  ifdft_set_error(errbuf, errbuf_size, "spin must be 0 for RKS or 1 for UKS");
  return IFDFT_IFXC_E_INVALID_ARGUMENT;
}

static int
ifdft_validate_response_vvars_args(
    const ifxc_dimensions_t *dims,
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
    const double *weighted_vvars,
    char *errbuf,
    size_t errbuf_size)
{
  if(dims == NULL || npoints == 0 || rho == NULL || sigma == NULL || tau == NULL
      || drho == NULL || dsigma == NULL || dtau == NULL || rho_gradient == NULL
      || drho_gradient == NULL || weights == NULL || feature_gradient == NULL
      || weighted_vvars == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing weighted response vvars input or output buffer");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  if(spin == 0){
    if(dims->rho != 1 || dims->sigma != 1 || dims->tau != 1){
      ifdft_set_error(errbuf, errbuf_size, "RKS response vvars require unpolarized MGGA dimensions");
      return IFDFT_IFXC_E_INVALID_ARGUMENT;
    }
    return IFDFT_IFXC_OK;
  }
  if(spin == 1){
    if(dims->rho != 2 || dims->sigma != 3 || dims->tau != 2){
      ifdft_set_error(errbuf, errbuf_size, "UKS response vvars require polarized MGGA dimensions");
      return IFDFT_IFXC_E_INVALID_ARGUMENT;
    }
    return IFDFT_IFXC_OK;
  }
  ifdft_set_error(errbuf, errbuf_size, "spin must be 0 for RKS or 1 for UKS");
  return IFDFT_IFXC_E_INVALID_ARGUMENT;
}

static int
ifdft_validate_weighted_vvars_args(
    const ifxc_dimensions_t *dims,
    int spin,
    size_t npoints,
    const double *rho,
    const double *sigma,
    const double *tau,
    const double *rho_gradient,
    const double *weights,
    const double *feature_gradient,
    const double *weighted_vvars,
    char *errbuf,
    size_t errbuf_size)
{
  if(dims == NULL || npoints == 0 || rho == NULL || sigma == NULL || tau == NULL
      || rho_gradient == NULL || weights == NULL || feature_gradient == NULL
      || weighted_vvars == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing weighted vvars input or output buffer");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  if(spin == 0){
    if(dims->rho != 1 || dims->sigma != 1 || dims->tau != 1){
      ifdft_set_error(errbuf, errbuf_size, "RKS weighted vvars require unpolarized MGGA dimensions");
      return IFDFT_IFXC_E_INVALID_ARGUMENT;
    }
    return IFDFT_IFXC_OK;
  }
  if(spin == 1){
    if(dims->rho != 2 || dims->sigma != 3 || dims->tau != 2){
      ifdft_set_error(errbuf, errbuf_size, "UKS weighted vvars require polarized MGGA dimensions");
      return IFDFT_IFXC_E_INVALID_ARGUMENT;
    }
    return IFDFT_IFXC_OK;
  }
  ifdft_set_error(errbuf, errbuf_size, "spin must be 0 for RKS or 1 for UKS");
  return IFDFT_IFXC_E_INVALID_ARGUMENT;
}

static void
ifdft_convert_weighted_vvars(
    int spin,
    size_t npoints,
    const double *rho_gradient,
    const double *weights,
    const double *d_rho,
    const double *d_sigma,
    const double *d_tau,
    double *weighted_vvars)
{
  size_t p;

  if(spin == 0){
    for(p = 0; p < npoints; ++p){
      const double w = weights[p];
      const double ds = d_sigma[p];
      weighted_vvars[0 * npoints + p] = w * d_rho[p];
      weighted_vvars[1 * npoints + p] = w * 2.0 * ds * rho_gradient[0 * npoints + p];
      weighted_vvars[2 * npoints + p] = w * 2.0 * ds * rho_gradient[1 * npoints + p];
      weighted_vvars[3 * npoints + p] = w * 2.0 * ds * rho_gradient[2 * npoints + p];
      weighted_vvars[4 * npoints + p] = w * d_tau[p];
    }
    return;
  }

  for(p = 0; p < npoints; ++p){
    const double w = weights[p];
    const double dr_a = d_rho[0 * npoints + p];
    const double dr_b = d_rho[1 * npoints + p];
    const double ds_aa = d_sigma[0 * npoints + p];
    const double ds_ab = d_sigma[1 * npoints + p];
    const double ds_bb = d_sigma[2 * npoints + p];
    const double dt_a = d_tau[0 * npoints + p];
    const double dt_b = d_tau[1 * npoints + p];
    const double *grad_a = rho_gradient;
    const double *grad_b = rho_gradient + 3 * npoints;
    double *wv_a = weighted_vvars;
    double *wv_b = weighted_vvars + 5 * npoints;

    wv_a[0 * npoints + p] = w * dr_a;
    wv_b[0 * npoints + p] = w * dr_b;
    wv_a[1 * npoints + p] =
        w * (2.0 * ds_aa * grad_a[0 * npoints + p] + ds_ab * grad_b[0 * npoints + p]);
    wv_a[2 * npoints + p] =
        w * (2.0 * ds_aa * grad_a[1 * npoints + p] + ds_ab * grad_b[1 * npoints + p]);
    wv_a[3 * npoints + p] =
        w * (2.0 * ds_aa * grad_a[2 * npoints + p] + ds_ab * grad_b[2 * npoints + p]);
    wv_b[1 * npoints + p] =
        w * (2.0 * ds_bb * grad_b[0 * npoints + p] + ds_ab * grad_a[0 * npoints + p]);
    wv_b[2 * npoints + p] =
        w * (2.0 * ds_bb * grad_b[1 * npoints + p] + ds_ab * grad_a[1 * npoints + p]);
    wv_b[3 * npoints + p] =
        w * (2.0 * ds_bb * grad_b[2 * npoints + p] + ds_ab * grad_a[2 * npoints + p]);
    wv_a[4 * npoints + p] = w * dt_a;
    wv_b[4 * npoints + p] = w * dt_b;
  }
}

static void
ifdft_contract_first_derivatives(
    size_t npoints,
    size_t nfeatures,
    size_t ncomponents,
    const double *feature_derivatives,
    const double *feature_gradient,
    double *contracted)
{
  size_t point;
  size_t component;
  size_t feature;

  for(component = 0; component < ncomponents; ++component){
    for(point = 0; point < npoints; ++point){
      double value = 0.0;
      for(feature = 0; feature < nfeatures; ++feature){
        value += feature_derivatives[
            ifdft_feature_deriv_index(point, feature, nfeatures, ncomponents, component)]
            * feature_gradient[feature];
      }
      contracted[component * npoints + point] = value;
    }
  }
}

static int
ifdft_eval_weighted_vvars_with_func(
    const ifxc_func_type *func,
    IFDFT_Workspace *workspace,
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
    size_t errbuf_size)
{
  ifxc_dimensions_t dims;
  double *d_rho = NULL;
  double *d_sigma = NULL;
  double *d_tau = NULL;
  int rc;
  ifxc_status status;

  if(func == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing evaluator function");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }

  status = ifxc_func_dimensions(func, &dims);
  if(status != IFXC_OK){
    return ifdft_status(status, errbuf, errbuf_size);
  }
  rc = ifdft_validate_weighted_vvars_args(
      &dims,
      spin,
      npoints,
      rho,
      sigma,
      tau,
      rho_gradient,
      weights,
      feature_gradient,
      weighted_vvars,
      errbuf,
      errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }

  if(workspace == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing evaluator workspace");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  rc = ifdft_workspace_require(&workspace->d_rho, &workspace->d_rho_capacity,
      dims.rho * npoints, "contracted d_rho", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  rc = ifdft_workspace_require(&workspace->d_sigma, &workspace->d_sigma_capacity,
      dims.sigma * npoints, "contracted d_sigma", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  rc = ifdft_workspace_require(&workspace->d_tau, &workspace->d_tau_capacity,
      dims.tau * npoints, "contracted d_tau", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  d_rho = workspace->d_rho;
  d_sigma = workspace->d_sigma;
  d_tau = workspace->d_tau;
  {
    ifxc_input input = {
      .npoints = npoints,
      .rho = rho,
      .sigma = sigma,
      .lapl = NULL,
      .tau = tau,
      .weights = NULL
    };
    status = ifxc_eval_first_derivatives_contracted(
        func, &input, feature_gradient, d_rho, d_sigma, d_tau);
    rc = ifdft_status(status, errbuf, errbuf_size);
  }
  if(rc == IFDFT_IFXC_OK){
    ifdft_convert_weighted_vvars(
        spin,
        npoints,
        rho_gradient,
        weights,
        d_rho,
        d_sigma,
        d_tau,
        weighted_vvars);
  }

  return rc;
}

static int
ifdft_validate_pyscf_mgga_input(
    int spin,
    size_t npoints,
    const double *rho_a,
    const double *rho_b,
    int ncomponents,
    char *errbuf,
    size_t errbuf_size)
{
  if(npoints == 0 || rho_a == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing PySCF MGGA rho input");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  if(ncomponents != 5 && ncomponents != 6){
    ifdft_set_error(errbuf, errbuf_size, "PySCF MGGA rho arrays must have 5 or 6 components");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  if(spin == 1 && rho_b == NULL){
    ifdft_set_error(errbuf, errbuf_size, "UKS PySCF MGGA conversion requires alpha and beta rho arrays");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  if(spin != 0 && spin != 1){
    ifdft_set_error(errbuf, errbuf_size, "spin must be 0 for RKS or 1 for UKS");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  return IFDFT_IFXC_OK;
}

static int
ifdft_prepare_pyscf_mgga_input(
    IFDFT_Workspace *workspace,
    int spin,
    size_t npoints,
    size_t source_npoints,
    size_t source_offset,
    const double *rho_a,
    const double *rho_b,
    int ncomponents,
    double **rho,
    double **sigma,
    double **tau,
    double **rho_gradient,
    char *errbuf,
    size_t errbuf_size)
{
  size_t point;
  size_t tau_component;
  int rc;
  int want_gradient = rho_gradient != NULL;

  rc = ifdft_validate_pyscf_mgga_input(spin, npoints, rho_a, rho_b, ncomponents, errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  if(source_offset > source_npoints ||
     npoints > source_npoints - source_offset){
    ifdft_set_error(errbuf, errbuf_size, "PySCF MGGA source slice is out of bounds");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  if(rho == NULL || sigma == NULL || tau == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing PySCF MGGA conversion output pointer");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  if(workspace == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing evaluator workspace");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }

  *rho = NULL;
  *sigma = NULL;
  *tau = NULL;
  if(want_gradient){
    *rho_gradient = NULL;
  }

  tau_component = (ncomponents == 6) ? 5 : 4;
  if(spin == 0){
    rc = ifdft_workspace_require(&workspace->rho, &workspace->rho_capacity,
        npoints, "rho", errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      return rc;
    }
    rc = ifdft_workspace_require(&workspace->sigma, &workspace->sigma_capacity,
        npoints, "sigma", errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      return rc;
    }
    rc = ifdft_workspace_require(&workspace->tau, &workspace->tau_capacity,
        npoints, "tau", errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      return rc;
    }
    if(want_gradient){
      rc = ifdft_workspace_require(
          &workspace->rho_gradient, &workspace->rho_gradient_capacity,
          3 * npoints, "rho-gradient", errbuf, errbuf_size);
      if(rc != IFDFT_IFXC_OK){
        return rc;
      }
    }
    *rho = workspace->rho;
    *sigma = workspace->sigma;
    *tau = workspace->tau;
    if(want_gradient){
      *rho_gradient = workspace->rho_gradient;
    }
    for(point = 0; point < npoints; ++point){
      const size_t source_point = source_offset + point;
      const double gx = rho_a[1 * source_npoints + source_point];
      const double gy = rho_a[2 * source_npoints + source_point];
      const double gz = rho_a[3 * source_npoints + source_point];
      (*rho)[point] = rho_a[source_point];
      (*sigma)[point] = gx * gx + gy * gy + gz * gz;
      (*tau)[point] = rho_a[tau_component * source_npoints + source_point];
      if(want_gradient){
        (*rho_gradient)[0 * npoints + point] = gx;
        (*rho_gradient)[1 * npoints + point] = gy;
        (*rho_gradient)[2 * npoints + point] = gz;
      }
    }
    return IFDFT_IFXC_OK;
  }

  rc = ifdft_workspace_require(&workspace->rho, &workspace->rho_capacity,
      2 * npoints, "rho", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  rc = ifdft_workspace_require(&workspace->sigma, &workspace->sigma_capacity,
      3 * npoints, "sigma", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  rc = ifdft_workspace_require(&workspace->tau, &workspace->tau_capacity,
      2 * npoints, "tau", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  if(want_gradient){
    rc = ifdft_workspace_require(
        &workspace->rho_gradient, &workspace->rho_gradient_capacity,
        6 * npoints, "rho-gradient", errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      return rc;
    }
  }
  *rho = workspace->rho;
  *sigma = workspace->sigma;
  *tau = workspace->tau;
  if(want_gradient){
    *rho_gradient = workspace->rho_gradient;
  }
  for(point = 0; point < npoints; ++point){
    const size_t source_point = source_offset + point;
    const double gax = rho_a[1 * source_npoints + source_point];
    const double gay = rho_a[2 * source_npoints + source_point];
    const double gaz = rho_a[3 * source_npoints + source_point];
    const double gbx = rho_b[1 * source_npoints + source_point];
    const double gby = rho_b[2 * source_npoints + source_point];
    const double gbz = rho_b[3 * source_npoints + source_point];
    (*rho)[0 * npoints + point] = rho_a[source_point];
    (*rho)[1 * npoints + point] = rho_b[source_point];
    (*sigma)[0 * npoints + point] = gax * gax + gay * gay + gaz * gaz;
    (*sigma)[1 * npoints + point] = gax * gbx + gay * gby + gaz * gbz;
    (*sigma)[2 * npoints + point] = gbx * gbx + gby * gby + gbz * gbz;
    (*tau)[0 * npoints + point] = rho_a[tau_component * source_npoints + source_point];
    (*tau)[1 * npoints + point] = rho_b[tau_component * source_npoints + source_point];
    if(want_gradient){
      (*rho_gradient)[0 * npoints + point] = gax;
      (*rho_gradient)[1 * npoints + point] = gay;
      (*rho_gradient)[2 * npoints + point] = gaz;
      (*rho_gradient)[3 * npoints + point] = gbx;
      (*rho_gradient)[4 * npoints + point] = gby;
      (*rho_gradient)[5 * npoints + point] = gbz;
    }
  }
  return IFDFT_IFXC_OK;
}

static void
ifdft_free_pyscf_mgga_input(
    IFDFT_Workspace *workspace,
    double *rho,
    double *sigma,
    double *tau,
    double *rho_gradient)
{
  if(workspace != NULL){
    return;
  }
  free(rho);
  free(sigma);
  free(tau);
  free(rho_gradient);
}

static void
ifdft_free_pyscf_mgga_direction(
    IFDFT_Workspace *workspace,
    double *drho,
    double *dsigma,
    double *dtau,
    double *drho_gradient)
{
  if(workspace != NULL){
    return;
  }
  free(drho);
  free(dsigma);
  free(dtau);
  free(drho_gradient);
}

static int
ifdft_prepare_pyscf_mgga_direction(
    IFDFT_Workspace *workspace,
    int spin,
    size_t npoints,
    size_t source_npoints,
    size_t source_start,
    const double *rho0_a,
    const double *rho0_b,
    const double *rho1_a,
    const double *rho1_b,
    int ncomponents,
    double **drho,
    double **dsigma,
    double **dtau,
    double **drho_gradient,
    char *errbuf,
    size_t errbuf_size)
{
  size_t point;
  size_t tau_component;
  int rc;
  int want_gradient = drho_gradient != NULL;

  if(source_start > source_npoints || npoints > source_npoints - source_start){
    ifdft_set_error(errbuf, errbuf_size, "invalid PySCF MGGA directional point slice");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  rc = ifdft_validate_pyscf_mgga_input(
      spin, source_npoints, rho0_a, rho0_b, ncomponents, errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  rc = ifdft_validate_pyscf_mgga_input(
      spin, source_npoints, rho1_a, rho1_b, ncomponents, errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  if(drho == NULL || dsigma == NULL || dtau == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing PySCF MGGA directional conversion output pointer");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  if(workspace == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing evaluator workspace");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }

  *drho = NULL;
  *dsigma = NULL;
  *dtau = NULL;
  if(want_gradient){
    *drho_gradient = NULL;
  }

  tau_component = (ncomponents == 6) ? 5 : 4;
  if(spin == 0){
    rc = ifdft_workspace_require(&workspace->drho, &workspace->drho_capacity,
        npoints, "direction-rho", errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      return rc;
    }
    rc = ifdft_workspace_require(&workspace->dsigma, &workspace->dsigma_capacity,
        npoints, "direction-sigma", errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      return rc;
    }
    rc = ifdft_workspace_require(&workspace->dtau, &workspace->dtau_capacity,
        npoints, "direction-tau", errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      return rc;
    }
    if(want_gradient){
      rc = ifdft_workspace_require(
          &workspace->drho_gradient, &workspace->drho_gradient_capacity,
          3 * npoints, "direction-rho-gradient", errbuf, errbuf_size);
      if(rc != IFDFT_IFXC_OK){
        return rc;
      }
    }
    *drho = workspace->drho;
    *dsigma = workspace->dsigma;
    *dtau = workspace->dtau;
    if(want_gradient){
      *drho_gradient = workspace->drho_gradient;
    }
    for(point = 0; point < npoints; ++point){
      size_t source_point = source_start + point;
      const double g0x = rho0_a[1 * source_npoints + source_point];
      const double g0y = rho0_a[2 * source_npoints + source_point];
      const double g0z = rho0_a[3 * source_npoints + source_point];
      const double g1x = rho1_a[1 * source_npoints + source_point];
      const double g1y = rho1_a[2 * source_npoints + source_point];
      const double g1z = rho1_a[3 * source_npoints + source_point];
      (*drho)[point] = rho1_a[source_point];
      (*dsigma)[point] = 2.0 * (g0x * g1x + g0y * g1y + g0z * g1z);
      (*dtau)[point] = rho1_a[tau_component * source_npoints + source_point];
      if(want_gradient){
        (*drho_gradient)[0 * npoints + point] = g1x;
        (*drho_gradient)[1 * npoints + point] = g1y;
        (*drho_gradient)[2 * npoints + point] = g1z;
      }
    }
    return IFDFT_IFXC_OK;
  }

  rc = ifdft_workspace_require(&workspace->drho, &workspace->drho_capacity,
      2 * npoints, "direction-rho", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  rc = ifdft_workspace_require(&workspace->dsigma, &workspace->dsigma_capacity,
      3 * npoints, "direction-sigma", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  rc = ifdft_workspace_require(&workspace->dtau, &workspace->dtau_capacity,
      2 * npoints, "direction-tau", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  if(want_gradient){
    rc = ifdft_workspace_require(
        &workspace->drho_gradient, &workspace->drho_gradient_capacity,
        6 * npoints, "direction-rho-gradient", errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      return rc;
    }
  }
  *drho = workspace->drho;
  *dsigma = workspace->dsigma;
  *dtau = workspace->dtau;
  if(want_gradient){
    *drho_gradient = workspace->drho_gradient;
  }
  for(point = 0; point < npoints; ++point){
    size_t source_point = source_start + point;
    const double g0ax = rho0_a[1 * source_npoints + source_point];
    const double g0ay = rho0_a[2 * source_npoints + source_point];
    const double g0az = rho0_a[3 * source_npoints + source_point];
    const double g0bx = rho0_b[1 * source_npoints + source_point];
    const double g0by = rho0_b[2 * source_npoints + source_point];
    const double g0bz = rho0_b[3 * source_npoints + source_point];
    const double g1ax = rho1_a[1 * source_npoints + source_point];
    const double g1ay = rho1_a[2 * source_npoints + source_point];
    const double g1az = rho1_a[3 * source_npoints + source_point];
    const double g1bx = rho1_b[1 * source_npoints + source_point];
    const double g1by = rho1_b[2 * source_npoints + source_point];
    const double g1bz = rho1_b[3 * source_npoints + source_point];
    (*drho)[0 * npoints + point] = rho1_a[source_point];
    (*drho)[1 * npoints + point] = rho1_b[source_point];
    (*dsigma)[0 * npoints + point] = 2.0 * (g0ax * g1ax + g0ay * g1ay + g0az * g1az);
    (*dsigma)[1 * npoints + point] =
        g1ax * g0bx + g1ay * g0by + g1az * g0bz
        + g0ax * g1bx + g0ay * g1by + g0az * g1bz;
    (*dsigma)[2 * npoints + point] = 2.0 * (g0bx * g1bx + g0by * g1by + g0bz * g1bz);
    (*dtau)[0 * npoints + point] = rho1_a[tau_component * source_npoints + source_point];
    (*dtau)[1 * npoints + point] = rho1_b[tau_component * source_npoints + source_point];
    if(want_gradient){
      (*drho_gradient)[0 * npoints + point] = g1ax;
      (*drho_gradient)[1 * npoints + point] = g1ay;
      (*drho_gradient)[2 * npoints + point] = g1az;
      (*drho_gradient)[3 * npoints + point] = g1bx;
      (*drho_gradient)[4 * npoints + point] = g1by;
      (*drho_gradient)[5 * npoints + point] = g1bz;
    }
  }
  return IFDFT_IFXC_OK;
}

static int
ifdft_eval_weighted_vvars_from_pyscf_mgga_with_func(
    const ifxc_func_type *func,
    IFDFT_Workspace *workspace,
    int spin,
    size_t npoints,
    const double *rho_a,
    const double *rho_b,
    int ncomponents,
    const double *weights,
    const double *feature_gradient,
    double *weighted_vvars,
    char *errbuf,
    size_t errbuf_size)
{
  double *rho = NULL;
  double *sigma = NULL;
  double *tau = NULL;
  double *rho_gradient = NULL;
  int rc;

  if(func == NULL || weights == NULL
      || feature_gradient == NULL || weighted_vvars == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing PySCF MGGA weighted vvars input or output buffer");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  rc = ifdft_prepare_pyscf_mgga_input(
      workspace,
      spin,
      npoints,
      npoints,
      0,
      rho_a,
      rho_b,
      ncomponents,
      &rho,
      &sigma,
      &tau,
      &rho_gradient,
      errbuf,
      errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }

  rc = ifdft_eval_weighted_vvars_with_func(
      func,
      workspace,
      spin,
      npoints,
      rho,
      sigma,
      tau,
      rho_gradient,
      weights,
      feature_gradient,
      weighted_vvars,
      errbuf,
      errbuf_size);

cleanup:
  ifdft_free_pyscf_mgga_input(workspace, rho, sigma, tau, rho_gradient);
  return rc;
}

static int
ifdft_eval_weighted_vvars_batch_from_pyscf_mgga_with_func(
    const ifxc_func_type *func,
    IFDFT_Workspace *workspace,
    int spin,
    size_t npoints,
    size_t source_npoints,
    size_t source_start,
    size_t ncoeff,
    const double *rho_a,
    const double *rho_b,
    int ncomponents,
    const double *weights,
    const double *feature_gradients,
    double *weighted_vvars_batch,
    char *errbuf,
    size_t errbuf_size)
{
  ifxc_dimensions_t dims;
  size_t nfeatures = 0;
  double *rho = NULL;
  double *sigma = NULL;
  double *tau = NULL;
  double *rho_gradient = NULL;
  double *d_rho = NULL;
  double *d_sigma = NULL;
  double *d_tau = NULL;
  double *feature_d_rho = NULL;
  double *feature_d_sigma = NULL;
  double *feature_d_tau = NULL;
  double *unit_weights = NULL;
  size_t coeff_idx;
  size_t point;
  size_t wv_size;
  int rc;
  ifxc_status status;

  if(func == NULL || ncoeff == 0 || weights == NULL
      || feature_gradients == NULL || weighted_vvars_batch == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing PySCF MGGA weighted vvars batch input or output buffer");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  if(workspace == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing evaluator workspace");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }

  status = ifxc_func_dimensions(func, &dims);
  if(status != IFXC_OK){
    return ifdft_status(status, errbuf, errbuf_size);
  }
  status = ifxc_nfeatures(func, &nfeatures);
  if(status != IFXC_OK){
    return ifdft_status(status, errbuf, errbuf_size);
  }

  rc = ifdft_prepare_pyscf_mgga_input(
      workspace,
      spin,
      npoints,
      source_npoints,
      source_start,
      rho_a,
      rho_b,
      ncomponents,
      &rho,
      &sigma,
      &tau,
      &rho_gradient,
      errbuf,
      errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }
  rc = ifdft_validate_weighted_vvars_args(
      &dims,
      spin,
      npoints,
      rho,
      sigma,
      tau,
      rho_gradient,
      weights,
      feature_gradients,
      weighted_vvars_batch,
      errbuf,
      errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }

  rc = ifdft_workspace_require(&workspace->d_rho, &workspace->d_rho_capacity,
      dims.rho * npoints, "contracted d_rho", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }
  rc = ifdft_workspace_require(&workspace->d_sigma, &workspace->d_sigma_capacity,
      dims.sigma * npoints, "contracted d_sigma", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }
  rc = ifdft_workspace_require(&workspace->d_tau, &workspace->d_tau_capacity,
      dims.tau * npoints, "contracted d_tau", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }
  rc = ifdft_workspace_require(&workspace->feature_d_rho, &workspace->feature_d_rho_capacity,
      npoints * nfeatures * dims.rho, "feature d_rho", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }
  rc = ifdft_workspace_require(&workspace->feature_d_sigma, &workspace->feature_d_sigma_capacity,
      npoints * nfeatures * dims.sigma, "feature d_sigma", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }
  rc = ifdft_workspace_require(&workspace->feature_d_tau, &workspace->feature_d_tau_capacity,
      npoints * nfeatures * dims.tau, "feature d_tau", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }
  rc = ifdft_workspace_require(&workspace->unit_weights, &workspace->unit_weights_capacity,
      npoints, "unit weights", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }

  d_rho = workspace->d_rho;
  d_sigma = workspace->d_sigma;
  d_tau = workspace->d_tau;
  feature_d_rho = workspace->feature_d_rho;
  feature_d_sigma = workspace->feature_d_sigma;
  feature_d_tau = workspace->feature_d_tau;
  unit_weights = workspace->unit_weights;
  for(point = 0; point < npoints; ++point){
    unit_weights[point] = 1.0;
  }

  rc = ifdft_eval_block_with_func(
      func,
      spin,
      npoints,
      rho,
      sigma,
      tau,
      unit_weights,
      IFDFT_IFXC_WANT_FIRST_DERIVATIVES,
      NULL,
      NULL,
      feature_d_rho,
      feature_d_sigma,
      feature_d_tau,
      NULL,
      NULL,
      NULL,
      NULL,
      NULL,
      NULL,
      errbuf,
      errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }

  wv_size = (spin == 0) ? 5 * npoints : 10 * npoints;
  for(coeff_idx = 0; coeff_idx < ncoeff; ++coeff_idx){
    const double *coeff = feature_gradients + coeff_idx * nfeatures;
    double *weighted_vvars = weighted_vvars_batch + coeff_idx * wv_size;
    ifdft_contract_first_derivatives(
        npoints, nfeatures, dims.rho, feature_d_rho, coeff, d_rho);
    ifdft_contract_first_derivatives(
        npoints, nfeatures, dims.sigma, feature_d_sigma, coeff, d_sigma);
    ifdft_contract_first_derivatives(
        npoints, nfeatures, dims.tau, feature_d_tau, coeff, d_tau);
    ifdft_convert_weighted_vvars(
        spin,
        npoints,
        rho_gradient,
        weights,
        d_rho,
        d_sigma,
        d_tau,
        weighted_vvars);
  }

cleanup:
  ifdft_free_pyscf_mgga_input(workspace, rho, sigma, tau, rho_gradient);
  return rc;
}

static int
ifdft_eval_features_from_pyscf_mgga_with_func(
    const ifxc_func_type *func,
    IFDFT_Workspace *workspace,
    int spin,
    size_t npoints,
    const double *rho_a,
    const double *rho_b,
    int ncomponents,
    const double *weights,
    double *features_integral,
    char *errbuf,
    size_t errbuf_size)
{
  double *rho = NULL;
  double *sigma = NULL;
  double *tau = NULL;
  int rc;

  if(func == NULL || weights == NULL || features_integral == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing PySCF MGGA feature input or output buffer");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  rc = ifdft_prepare_pyscf_mgga_input(
      workspace,
      spin,
      npoints,
      npoints,
      0,
      rho_a,
      rho_b,
      ncomponents,
      &rho,
      &sigma,
      &tau,
      NULL,
      errbuf,
      errbuf_size);
  if(rc == IFDFT_IFXC_OK){
    rc = ifdft_eval_block_with_func(
        func,
        spin,
        npoints,
        rho,
        sigma,
        tau,
        weights,
        IFDFT_IFXC_WANT_INTEGRAL_FEATURES,
        features_integral,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        errbuf,
        errbuf_size);
  }
  ifdft_free_pyscf_mgga_input(workspace, rho, sigma, tau, NULL);
  return rc;
}

static size_t
ifdft_parallel_worker_count(
    const IFDFT_IFXC_Evaluator *evaluator,
    size_t npoints)
{
#ifndef _OPENMP
  (void)evaluator;
  (void)npoints;
  return 1;
#else
  size_t useful_workers;
  size_t requested_workers;

  useful_workers = npoints / IFDFT_MIN_POINTS_PER_WORKER;
  if(useful_workers < 1){
    useful_workers = 1;
  }
  requested_workers = (evaluator != NULL && evaluator->requested_workers > 0)
      ? (size_t)evaluator->requested_workers
      : 1;
  return requested_workers < useful_workers ? requested_workers : useful_workers;
#endif
}

static int
ifdft_parallel_eval_weighted_vvars_from_pyscf_mgga(
    IFDFT_IFXC_Evaluator *evaluator,
    int spin,
    size_t npoints,
    const double *rho_a,
    const double *rho_b,
    int ncomponents,
    const double *weights,
    const double *feature_gradient,
    double *weighted_vvars,
    char *errbuf,
    size_t errbuf_size)
{
  size_t nworkers;
  size_t worker_index;
  size_t output_components;
  int *statuses = NULL;
  int rc;

  if(evaluator == NULL || weights == NULL || feature_gradient == NULL ||
     weighted_vvars == NULL){
    ifdft_set_error(errbuf, errbuf_size,
                    "missing parallel PySCF MGGA weighted-vvar input");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  rc = ifdft_validate_pyscf_mgga_input(
      spin, npoints, rho_a, rho_b, ncomponents, errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }

  nworkers = ifdft_parallel_worker_count(evaluator, npoints);
  rc = ifdft_evaluator_require_workers(
      evaluator, nworkers, errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  {
    ifxc_func_type *func = NULL;
    rc = ifdft_evaluator_get_worker_func(
        evaluator, 0, spin, &func, errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      return rc;
    }
    if(nworkers == 1){
      return ifdft_eval_weighted_vvars_from_pyscf_mgga_with_func(
          func, &evaluator->workers[0].workspace, spin, npoints,
          rho_a, rho_b, ncomponents, weights, feature_gradient,
          weighted_vvars, errbuf, errbuf_size);
    }
  }
  output_components = (spin == 0) ? 5 : 10;
  statuses = (int *)calloc(nworkers, sizeof(int));
  if(statuses == NULL){
    ifdft_set_error(errbuf, errbuf_size, "failed to allocate worker statuses");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }

  for(worker_index = 0; worker_index < nworkers; ++worker_index){
    ifxc_func_type *func = NULL;
    size_t start = npoints * worker_index / nworkers;
    size_t end = npoints * (worker_index + 1) / nworkers;
    size_t chunk_size = end - start;
    rc = ifdft_evaluator_get_worker_func(
        evaluator, worker_index, spin, &func, errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      free(statuses);
      return rc;
    }
    rc = ifdft_workspace_require(
        &evaluator->workers[worker_index].workspace.weighted_vvars,
        &evaluator->workers[worker_index].workspace.weighted_vvars_capacity,
        output_components * chunk_size,
        "worker weighted vvars", errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      free(statuses);
      return rc;
    }
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1) num_threads((int)nworkers)
#endif
  for(worker_index = 0; worker_index < nworkers; ++worker_index){
    IFDFT_IFXC_Worker *worker = &evaluator->workers[worker_index];
    ifxc_func_type *func = &worker->funcs[ifdft_spin_index(spin)];
    size_t start = npoints * worker_index / nworkers;
    size_t end = npoints * (worker_index + 1) / nworkers;
    size_t chunk_size = end - start;
    double *rho = NULL;
    double *sigma = NULL;
    double *tau = NULL;
    double *rho_gradient = NULL;
    double *local_output = worker->workspace.weighted_vvars;
    size_t component;

    statuses[worker_index] = ifdft_prepare_pyscf_mgga_input(
        &worker->workspace, spin, chunk_size, npoints, start,
        rho_a, rho_b, ncomponents, &rho, &sigma, &tau, &rho_gradient,
        NULL, 0);
    if(statuses[worker_index] == IFDFT_IFXC_OK){
      statuses[worker_index] = ifdft_eval_weighted_vvars_with_func(
          func, &worker->workspace, spin, chunk_size,
          rho, sigma, tau, rho_gradient, weights + start,
          feature_gradient, local_output, NULL, 0);
    }
    if(statuses[worker_index] == IFDFT_IFXC_OK){
      for(component = 0; component < output_components; ++component){
        memcpy(weighted_vvars + component * npoints + start,
               local_output + component * chunk_size,
               chunk_size * sizeof(double));
      }
    }
  }

  rc = IFDFT_IFXC_OK;
  for(worker_index = 0; worker_index < nworkers; ++worker_index){
    if(statuses[worker_index] != IFDFT_IFXC_OK){
      rc = statuses[worker_index];
      ifdft_set_error(errbuf, errbuf_size,
                      "parallel PySCF MGGA weighted-vvar worker failed");
      break;
    }
  }
  free(statuses);
  return rc;
}

static int
ifdft_parallel_eval_features_from_pyscf_mgga(
    IFDFT_IFXC_Evaluator *evaluator,
    int spin,
    size_t npoints,
    const double *rho_a,
    const double *rho_b,
    int ncomponents,
    const double *weights,
    double *features_integral,
    char *errbuf,
    size_t errbuf_size)
{
  size_t nworkers;
  size_t nfeatures = 0;
  size_t worker_index;
  int *statuses = NULL;
  int rc;

  if(evaluator == NULL || weights == NULL || features_integral == NULL){
    ifdft_set_error(errbuf, errbuf_size,
                    "missing parallel PySCF MGGA feature input");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  rc = ifdft_validate_pyscf_mgga_input(
      spin, npoints, rho_a, rho_b, ncomponents, errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  nworkers = ifdft_parallel_worker_count(evaluator, npoints);
  rc = ifdft_evaluator_require_workers(
      evaluator, nworkers, errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  statuses = (int *)calloc(nworkers, sizeof(int));
  if(statuses == NULL){
    ifdft_set_error(errbuf, errbuf_size, "failed to allocate worker statuses");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }

  for(worker_index = 0; worker_index < nworkers; ++worker_index){
    ifxc_func_type *func = NULL;
    ifxc_status status;
    rc = ifdft_evaluator_get_worker_func(
        evaluator, worker_index, spin, &func, errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      free(statuses);
      return rc;
    }
    if(worker_index == 0){
      status = ifxc_nfeatures(func, &nfeatures);
      if(status != IFXC_OK){
        free(statuses);
        return ifdft_status(status, errbuf, errbuf_size);
      }
    }
    rc = ifdft_workspace_require(
        &evaluator->workers[worker_index].workspace.features,
        &evaluator->workers[worker_index].workspace.features_capacity,
        nfeatures, "worker feature integral", errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      free(statuses);
      return rc;
    }
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1) num_threads((int)nworkers)
#endif
  for(worker_index = 0; worker_index < nworkers; ++worker_index){
    IFDFT_IFXC_Worker *worker = &evaluator->workers[worker_index];
    ifxc_func_type *func = &worker->funcs[ifdft_spin_index(spin)];
    size_t start = npoints * worker_index / nworkers;
    size_t end = npoints * (worker_index + 1) / nworkers;
    size_t chunk_size = end - start;
    double *rho = NULL;
    double *sigma = NULL;
    double *tau = NULL;

    statuses[worker_index] = ifdft_prepare_pyscf_mgga_input(
        &worker->workspace, spin, chunk_size, npoints, start,
        rho_a, rho_b, ncomponents, &rho, &sigma, &tau, NULL, NULL, 0);
    if(statuses[worker_index] == IFDFT_IFXC_OK){
      statuses[worker_index] = ifdft_eval_block_with_func(
          func, spin, chunk_size, rho, sigma, tau, weights + start,
          IFDFT_IFXC_WANT_INTEGRAL_FEATURES, worker->workspace.features,
          NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
          NULL, 0);
    }
  }

  memset(features_integral, 0, nfeatures * sizeof(double));
  rc = IFDFT_IFXC_OK;
  for(worker_index = 0; worker_index < nworkers; ++worker_index){
    size_t feature;
    if(statuses[worker_index] != IFDFT_IFXC_OK){
      rc = statuses[worker_index];
      ifdft_set_error(errbuf, errbuf_size,
                      "parallel PySCF MGGA feature worker failed");
      break;
    }
    for(feature = 0; feature < nfeatures; ++feature){
      features_integral[feature] +=
          evaluator->workers[worker_index].workspace.features[feature];
    }
  }
  free(statuses);
  return rc;
}

static int
ifdft_eval_feature_directional_from_pyscf_mgga_with_func(
    const ifxc_func_type *func,
    IFDFT_Workspace *workspace,
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
    size_t errbuf_size)
{
  double *rho = NULL;
  double *sigma = NULL;
  double *tau = NULL;
  double *drho = NULL;
  double *dsigma = NULL;
  double *dtau = NULL;
  int rc;

  if(func == NULL || weights == NULL || dfeatures == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing PySCF MGGA directional input or output buffer");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  rc = ifdft_prepare_pyscf_mgga_input(
      workspace,
      spin,
      npoints,
      npoints,
      0,
      rho0_a,
      rho0_b,
      ncomponents,
      &rho,
      &sigma,
      &tau,
      NULL,
      errbuf,
      errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }
  rc = ifdft_prepare_pyscf_mgga_direction(
      workspace,
      spin,
      npoints,
      npoints,
      0,
      rho0_a,
      rho0_b,
      rho1_a,
      rho1_b,
      ncomponents,
      &drho,
      &dsigma,
      &dtau,
      NULL,
      errbuf,
      errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }
  rc = ifdft_eval_feature_directional_derivative_with_func(
      func,
      workspace,
      spin,
      npoints,
      rho,
      sigma,
      tau,
      drho,
      dsigma,
      dtau,
      weights,
      dfeatures,
      errbuf,
      errbuf_size);

cleanup:
  ifdft_free_pyscf_mgga_input(workspace, rho, sigma, tau, NULL);
  ifdft_free_pyscf_mgga_direction(workspace, drho, dsigma, dtau, NULL);
  return rc;
}

static int
ifdft_eval_weighted_response_from_pyscf_mgga_with_func(
    const ifxc_func_type *func,
    IFDFT_Workspace *workspace,
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
    size_t errbuf_size)
{
  double *rho = NULL;
  double *sigma = NULL;
  double *tau = NULL;
  double *rho_gradient = NULL;
  double *drho = NULL;
  double *dsigma = NULL;
  double *dtau = NULL;
  double *drho_gradient = NULL;
  int rc;

  if(func == NULL || weights == NULL || feature_gradient == NULL || weighted_vvars == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing PySCF MGGA response input or output buffer");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  rc = ifdft_prepare_pyscf_mgga_input(
      workspace,
      spin,
      npoints,
      npoints,
      0,
      rho0_a,
      rho0_b,
      ncomponents,
      &rho,
      &sigma,
      &tau,
      &rho_gradient,
      errbuf,
      errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }
  rc = ifdft_prepare_pyscf_mgga_direction(
      workspace,
      spin,
      npoints,
      npoints,
      0,
      rho0_a,
      rho0_b,
      rho1_a,
      rho1_b,
      ncomponents,
      &drho,
      &dsigma,
      &dtau,
      &drho_gradient,
      errbuf,
      errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }
  rc = ifdft_eval_weighted_response_vvars_with_func(
      func,
      workspace,
      spin,
      npoints,
      rho,
      sigma,
      tau,
      drho,
      dsigma,
      dtau,
      rho_gradient,
      drho_gradient,
      weights,
      feature_gradient,
      weighted_vvars,
      errbuf,
      errbuf_size);

cleanup:
  ifdft_free_pyscf_mgga_input(workspace, rho, sigma, tau, rho_gradient);
  ifdft_free_pyscf_mgga_direction(workspace, drho, dsigma, dtau, drho_gradient);
  return rc;
}

static void
ifdft_accumulate_feature_directional(
    size_t npoints,
    size_t nfeatures,
    size_t rho_dim,
    size_t sigma_dim,
    size_t tau_dim,
    const double *d_rho,
    const double *d_sigma,
    const double *d_tau,
    const double *drho,
    const double *dsigma,
    const double *dtau,
    double *dfeatures)
{
  size_t feature;
  size_t point;

  for(feature = 0; feature < nfeatures; ++feature){
    dfeatures[feature] = 0.0;
  }
  for(point = 0; point < npoints; ++point){
    for(feature = 0; feature < nfeatures; ++feature){
      double value = 0.0;
      size_t component;
      for(component = 0; component < rho_dim; ++component){
        value += d_rho[ifdft_feature_deriv_index(point, feature, nfeatures, rho_dim, component)]
            * drho[component * npoints + point];
      }
      for(component = 0; component < sigma_dim; ++component){
        value += d_sigma[ifdft_feature_deriv_index(point, feature, nfeatures, sigma_dim, component)]
            * dsigma[component * npoints + point];
      }
      for(component = 0; component < tau_dim; ++component){
        value += d_tau[ifdft_feature_deriv_index(point, feature, nfeatures, tau_dim, component)]
            * dtau[component * npoints + point];
      }
      dfeatures[feature] += value;
    }
  }
}

static int
ifdft_eval_feature_directional_derivative_with_func(
    const ifxc_func_type *func,
    IFDFT_Workspace *workspace,
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
    size_t errbuf_size)
{
  ifxc_dimensions_t dims;
  size_t nfeatures = 0;
  double *d_rho = NULL;
  double *d_sigma = NULL;
  double *d_tau = NULL;
  int rc;
  ifxc_status status;

  if(func == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing evaluator function");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  status = ifxc_func_dimensions(func, &dims);
  if(status != IFXC_OK){
    return ifdft_status(status, errbuf, errbuf_size);
  }
  status = ifxc_nfeatures(func, &nfeatures);
  if(status != IFXC_OK){
    return ifdft_status(status, errbuf, errbuf_size);
  }
  rc = ifdft_validate_directional_args(
      &dims,
      spin,
      npoints,
      rho,
      sigma,
      tau,
      drho,
      dsigma,
      dtau,
      weights,
      dfeatures,
      errbuf,
      errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }

  if(workspace == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing evaluator workspace");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  rc = ifdft_workspace_require(&workspace->feature_d_rho, &workspace->feature_d_rho_capacity,
      npoints * nfeatures * dims.rho, "feature d_rho", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  rc = ifdft_workspace_require(&workspace->feature_d_sigma, &workspace->feature_d_sigma_capacity,
      npoints * nfeatures * dims.sigma, "feature d_sigma", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  rc = ifdft_workspace_require(&workspace->feature_d_tau, &workspace->feature_d_tau_capacity,
      npoints * nfeatures * dims.tau, "feature d_tau", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  d_rho = workspace->feature_d_rho;
  d_sigma = workspace->feature_d_sigma;
  d_tau = workspace->feature_d_tau;

  rc = ifdft_eval_block_with_func(
      func,
      spin,
      npoints,
      rho,
      sigma,
      tau,
      weights,
      IFDFT_IFXC_WANT_FIRST_DERIVATIVES,
      NULL,
      NULL,
      d_rho,
      d_sigma,
      d_tau,
      NULL,
      NULL,
      NULL,
      NULL,
      NULL,
      NULL,
      errbuf,
      errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }

  ifdft_accumulate_feature_directional(
      npoints,
      nfeatures,
      dims.rho,
      dims.sigma,
      dims.tau,
      d_rho,
      d_sigma,
      d_tau,
      drho,
      dsigma,
      dtau,
      dfeatures);

  return IFDFT_IFXC_OK;
}

static void
ifdft_convert_response_vvars_rks(
    size_t npoints,
    size_t nfeatures,
    const double *drho,
    const double *dsigma,
    const double *dtau,
    const double *rho_gradient,
    const double *drho_gradient,
    const double *feature_gradient,
    const double *d_sigma,
    const double *dd_rho_rho,
    const double *dd_rho_sigma,
    const double *dd_rho_tau,
    const double *dd_sigma_sigma,
    const double *dd_sigma_tau,
    const double *dd_tau_tau,
    double *weighted_vvars)
{
  size_t point;
  size_t feature;

  for(point = 0; point < npoints; ++point){
    double first_sigma = 0.0;
    double h_rr = 0.0;
    double h_rs = 0.0;
    double h_rt = 0.0;
    double h_ss = 0.0;
    double h_st = 0.0;
    double h_tt = 0.0;
    double dv_rho;
    double dv_sigma;
    double dv_tau;
    size_t idx;

    for(feature = 0; feature < nfeatures; ++feature){
      const double coeff = feature_gradient[feature];
      idx = ifdft_feature_deriv_index(point, feature, nfeatures, 1, 0);
      first_sigma += d_sigma[idx] * coeff;
      h_rr += dd_rho_rho[idx] * coeff;
      h_rs += dd_rho_sigma[idx] * coeff;
      h_rt += dd_rho_tau[idx] * coeff;
      h_ss += dd_sigma_sigma[idx] * coeff;
      h_st += dd_sigma_tau[idx] * coeff;
      h_tt += dd_tau_tau[idx] * coeff;
    }

    dv_rho = h_rr * drho[point] + h_rs * dsigma[point] + h_rt * dtau[point];
    dv_sigma = h_rs * drho[point] + h_ss * dsigma[point] + h_st * dtau[point];
    dv_tau = h_rt * drho[point] + h_st * dsigma[point] + h_tt * dtau[point];

    weighted_vvars[0 * npoints + point] = dv_rho;
    weighted_vvars[1 * npoints + point] =
        2.0 * (dv_sigma * rho_gradient[0 * npoints + point]
            + first_sigma * drho_gradient[0 * npoints + point]);
    weighted_vvars[2 * npoints + point] =
        2.0 * (dv_sigma * rho_gradient[1 * npoints + point]
            + first_sigma * drho_gradient[1 * npoints + point]);
    weighted_vvars[3 * npoints + point] =
        2.0 * (dv_sigma * rho_gradient[2 * npoints + point]
            + first_sigma * drho_gradient[2 * npoints + point]);
    weighted_vvars[4 * npoints + point] = dv_tau;
  }
}

static void
ifdft_convert_response_vvars_uks(
    size_t npoints,
    size_t nfeatures,
    const double *drho,
    const double *dsigma,
    const double *dtau,
    const double *rho_gradient,
    const double *drho_gradient,
    const double *feature_gradient,
    const double *d_sigma,
    const double *dd_rho_rho,
    const double *dd_rho_sigma,
    const double *dd_rho_tau,
    const double *dd_sigma_sigma,
    const double *dd_sigma_tau,
    const double *dd_tau_tau,
    double *weighted_vvars)
{
  size_t point;
  size_t feature;

  for(point = 0; point < npoints; ++point){
    double first_sigma[3] = {0.0, 0.0, 0.0};
    double h_rr[3] = {0.0, 0.0, 0.0};
    double h_rs[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double h_rt[4] = {0.0, 0.0, 0.0, 0.0};
    double h_ss[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double h_st[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double h_tt[3] = {0.0, 0.0, 0.0};
    double dv_rho[2];
    double dv_sigma[3];
    double dv_tau[2];
    const double dr_a = drho[0 * npoints + point];
    const double dr_b = drho[1 * npoints + point];
    const double ds_aa = dsigma[0 * npoints + point];
    const double ds_ab = dsigma[1 * npoints + point];
    const double ds_bb = dsigma[2 * npoints + point];
    const double dt_a = dtau[0 * npoints + point];
    const double dt_b = dtau[1 * npoints + point];
    const double *grad_a = rho_gradient;
    const double *grad_b = rho_gradient + 3 * npoints;
    const double *dgrad_a = drho_gradient;
    const double *dgrad_b = drho_gradient + 3 * npoints;
    double *wv_a = weighted_vvars;
    double *wv_b = weighted_vvars + 5 * npoints;

    for(feature = 0; feature < nfeatures; ++feature){
      const double coeff = feature_gradient[feature];
      size_t component;
      for(component = 0; component < 3; ++component){
        first_sigma[component] +=
            d_sigma[ifdft_feature_deriv_index(point, feature, nfeatures, 3, component)] * coeff;
        h_rr[component] +=
            dd_rho_rho[ifdft_feature_deriv_index(point, feature, nfeatures, 3, component)] * coeff;
        h_ss[component] +=
            dd_sigma_sigma[ifdft_feature_deriv_index(point, feature, nfeatures, 6, component)] * coeff;
        h_tt[component] +=
            dd_tau_tau[ifdft_feature_deriv_index(point, feature, nfeatures, 3, component)] * coeff;
      }
      for(component = 0; component < 6; ++component){
        h_rs[component] +=
            dd_rho_sigma[ifdft_feature_deriv_index(point, feature, nfeatures, 6, component)] * coeff;
        h_st[component] +=
            dd_sigma_tau[ifdft_feature_deriv_index(point, feature, nfeatures, 6, component)] * coeff;
      }
      for(component = 0; component < 4; ++component){
        h_rt[component] +=
            dd_rho_tau[ifdft_feature_deriv_index(point, feature, nfeatures, 4, component)] * coeff;
      }
      for(component = 3; component < 6; ++component){
        h_ss[component] +=
            dd_sigma_sigma[ifdft_feature_deriv_index(point, feature, nfeatures, 6, component)] * coeff;
      }
    }

    dv_rho[0] = h_rr[0] * dr_a + h_rr[1] * dr_b
        + h_rs[0] * ds_aa + h_rs[1] * ds_ab + h_rs[2] * ds_bb
        + h_rt[0] * dt_a + h_rt[1] * dt_b;
    dv_rho[1] = h_rr[1] * dr_a + h_rr[2] * dr_b
        + h_rs[3] * ds_aa + h_rs[4] * ds_ab + h_rs[5] * ds_bb
        + h_rt[2] * dt_a + h_rt[3] * dt_b;

    dv_sigma[0] = h_rs[0] * dr_a + h_rs[3] * dr_b
        + h_ss[0] * ds_aa + h_ss[1] * ds_ab + h_ss[2] * ds_bb
        + h_st[0] * dt_a + h_st[1] * dt_b;
    dv_sigma[1] = h_rs[1] * dr_a + h_rs[4] * dr_b
        + h_ss[1] * ds_aa + h_ss[3] * ds_ab + h_ss[4] * ds_bb
        + h_st[2] * dt_a + h_st[3] * dt_b;
    dv_sigma[2] = h_rs[2] * dr_a + h_rs[5] * dr_b
        + h_ss[2] * ds_aa + h_ss[4] * ds_ab + h_ss[5] * ds_bb
        + h_st[4] * dt_a + h_st[5] * dt_b;

    dv_tau[0] = h_rt[0] * dr_a + h_rt[2] * dr_b
        + h_st[0] * ds_aa + h_st[2] * ds_ab + h_st[4] * ds_bb
        + h_tt[0] * dt_a + h_tt[1] * dt_b;
    dv_tau[1] = h_rt[1] * dr_a + h_rt[3] * dr_b
        + h_st[1] * ds_aa + h_st[3] * ds_ab + h_st[5] * ds_bb
        + h_tt[1] * dt_a + h_tt[2] * dt_b;

    wv_a[0 * npoints + point] = dv_rho[0];
    wv_b[0 * npoints + point] = dv_rho[1];
    wv_a[1 * npoints + point] =
        2.0 * dv_sigma[0] * grad_a[0 * npoints + point]
        + dv_sigma[1] * grad_b[0 * npoints + point]
        + 2.0 * first_sigma[0] * dgrad_a[0 * npoints + point]
        + first_sigma[1] * dgrad_b[0 * npoints + point];
    wv_a[2 * npoints + point] =
        2.0 * dv_sigma[0] * grad_a[1 * npoints + point]
        + dv_sigma[1] * grad_b[1 * npoints + point]
        + 2.0 * first_sigma[0] * dgrad_a[1 * npoints + point]
        + first_sigma[1] * dgrad_b[1 * npoints + point];
    wv_a[3 * npoints + point] =
        2.0 * dv_sigma[0] * grad_a[2 * npoints + point]
        + dv_sigma[1] * grad_b[2 * npoints + point]
        + 2.0 * first_sigma[0] * dgrad_a[2 * npoints + point]
        + first_sigma[1] * dgrad_b[2 * npoints + point];
    wv_b[1 * npoints + point] =
        2.0 * dv_sigma[2] * grad_b[0 * npoints + point]
        + dv_sigma[1] * grad_a[0 * npoints + point]
        + 2.0 * first_sigma[2] * dgrad_b[0 * npoints + point]
        + first_sigma[1] * dgrad_a[0 * npoints + point];
    wv_b[2 * npoints + point] =
        2.0 * dv_sigma[2] * grad_b[1 * npoints + point]
        + dv_sigma[1] * grad_a[1 * npoints + point]
        + 2.0 * first_sigma[2] * dgrad_b[1 * npoints + point]
        + first_sigma[1] * dgrad_a[1 * npoints + point];
    wv_b[3 * npoints + point] =
        2.0 * dv_sigma[2] * grad_b[2 * npoints + point]
        + dv_sigma[1] * grad_a[2 * npoints + point]
        + 2.0 * first_sigma[2] * dgrad_b[2 * npoints + point]
        + first_sigma[1] * dgrad_a[2 * npoints + point];
    wv_a[4 * npoints + point] = dv_tau[0];
    wv_b[4 * npoints + point] = dv_tau[1];
  }
}

static int
ifdft_eval_response_batch_from_pyscf_mgga_with_func(
    const ifxc_func_type *func,
    IFDFT_Workspace *workspace,
    int spin,
    size_t npoints,
    size_t source_npoints,
    size_t source_start,
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
    size_t errbuf_size)
{
  ifxc_dimensions_t dims;
  size_t nfeatures = 0;
  double *rho = NULL;
  double *sigma = NULL;
  double *tau = NULL;
  double *rho_gradient = NULL;
  double *drho = NULL;
  double *dsigma = NULL;
  double *dtau = NULL;
  double *drho_gradient = NULL;
  double *d_rho = NULL;
  double *d_sigma = NULL;
  double *d_tau = NULL;
  double *dd_rho_rho = NULL;
  double *dd_rho_sigma = NULL;
  double *dd_rho_tau = NULL;
  double *dd_sigma_sigma = NULL;
  double *dd_sigma_tau = NULL;
  double *dd_tau_tau = NULL;
  size_t trial;
  size_t rho_stride;
  size_t wv_size;
  int rc;
  ifxc_status status;

  if(func == NULL || ntrial == 0 || weights == NULL || feature_gradient == NULL
      || weighted_kernel_vvars == NULL || dfeatures == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing PySCF MGGA response batch input or output buffer");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  if(workspace == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing evaluator workspace");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }

  status = ifxc_func_dimensions(func, &dims);
  if(status != IFXC_OK){
    return ifdft_status(status, errbuf, errbuf_size);
  }
  status = ifxc_nfeatures(func, &nfeatures);
  if(status != IFXC_OK){
    return ifdft_status(status, errbuf, errbuf_size);
  }

  rc = ifdft_prepare_pyscf_mgga_input(
      workspace,
      spin,
      npoints,
      source_npoints,
      source_start,
      rho0_a,
      rho0_b,
      ncomponents,
      &rho,
      &sigma,
      &tau,
      &rho_gradient,
      errbuf,
      errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }

  rc = ifdft_workspace_require(&workspace->feature_d_rho, &workspace->feature_d_rho_capacity,
      npoints * nfeatures * dims.rho, "feature d_rho", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }
  rc = ifdft_workspace_require(&workspace->feature_d_sigma, &workspace->feature_d_sigma_capacity,
      npoints * nfeatures * dims.sigma, "feature d_sigma", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }
  rc = ifdft_workspace_require(&workspace->feature_d_tau, &workspace->feature_d_tau_capacity,
      npoints * nfeatures * dims.tau, "feature d_tau", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }
  rc = ifdft_workspace_require(&workspace->dd_rho_rho, &workspace->dd_rho_rho_capacity,
      npoints * nfeatures * ifdft_sym_count(dims.rho), "dd_rho_rho", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }
  rc = ifdft_workspace_require(&workspace->dd_rho_sigma, &workspace->dd_rho_sigma_capacity,
      npoints * nfeatures * dims.rho * dims.sigma, "dd_rho_sigma", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }
  rc = ifdft_workspace_require(&workspace->dd_rho_tau, &workspace->dd_rho_tau_capacity,
      npoints * nfeatures * dims.rho * dims.tau, "dd_rho_tau", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }
  rc = ifdft_workspace_require(&workspace->dd_sigma_sigma, &workspace->dd_sigma_sigma_capacity,
      npoints * nfeatures * ifdft_sym_count(dims.sigma), "dd_sigma_sigma", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }
  rc = ifdft_workspace_require(&workspace->dd_sigma_tau, &workspace->dd_sigma_tau_capacity,
      npoints * nfeatures * dims.sigma * dims.tau, "dd_sigma_tau", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }
  rc = ifdft_workspace_require(&workspace->dd_tau_tau, &workspace->dd_tau_tau_capacity,
      npoints * nfeatures * ifdft_sym_count(dims.tau), "dd_tau_tau", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }

  d_rho = workspace->feature_d_rho;
  d_sigma = workspace->feature_d_sigma;
  d_tau = workspace->feature_d_tau;
  dd_rho_rho = workspace->dd_rho_rho;
  dd_rho_sigma = workspace->dd_rho_sigma;
  dd_rho_tau = workspace->dd_rho_tau;
  dd_sigma_sigma = workspace->dd_sigma_sigma;
  dd_sigma_tau = workspace->dd_sigma_tau;
  dd_tau_tau = workspace->dd_tau_tau;

  rc = ifdft_eval_block_with_func(
      func,
      spin,
      npoints,
      rho,
      sigma,
      tau,
      weights,
      IFDFT_IFXC_WANT_FIRST_DERIVATIVES | IFDFT_IFXC_WANT_SECOND_DERIVATIVES,
      NULL,
      NULL,
      d_rho,
      d_sigma,
      d_tau,
      dd_rho_rho,
      dd_rho_sigma,
      dd_rho_tau,
      dd_sigma_sigma,
      dd_sigma_tau,
      dd_tau_tau,
      errbuf,
      errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    goto cleanup;
  }

  rho_stride = (size_t)ncomponents * source_npoints;
  wv_size = (spin == 0) ? 5 * npoints : 10 * npoints;
  for(trial = 0; trial < ntrial; ++trial){
    const double *rho1_a = rho1_a_batch + trial * rho_stride;
    const double *rho1_b = (spin == 0) ? NULL : rho1_b_batch + trial * rho_stride;
    double *trial_dfeatures = dfeatures + trial * nfeatures;
    double *trial_wv = weighted_kernel_vvars + trial * wv_size;
    rc = ifdft_prepare_pyscf_mgga_direction(
        workspace,
        spin,
        npoints,
        source_npoints,
        source_start,
        rho0_a,
        rho0_b,
        rho1_a,
        rho1_b,
        ncomponents,
        &drho,
        &dsigma,
        &dtau,
        &drho_gradient,
        errbuf,
        errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      goto cleanup;
    }
    rc = ifdft_validate_response_vvars_args(
        &dims,
        spin,
        npoints,
        rho,
        sigma,
        tau,
        drho,
        dsigma,
        dtau,
        rho_gradient,
        drho_gradient,
        weights,
        feature_gradient,
        trial_wv,
        errbuf,
        errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      goto cleanup;
    }
    ifdft_accumulate_feature_directional(
        npoints,
        nfeatures,
        dims.rho,
        dims.sigma,
        dims.tau,
        d_rho,
        d_sigma,
        d_tau,
        drho,
        dsigma,
        dtau,
        trial_dfeatures);
    if(spin == 0){
      ifdft_convert_response_vvars_rks(
          npoints,
          nfeatures,
          drho,
          dsigma,
          dtau,
          rho_gradient,
          drho_gradient,
          feature_gradient,
          d_sigma,
          dd_rho_rho,
          dd_rho_sigma,
          dd_rho_tau,
          dd_sigma_sigma,
          dd_sigma_tau,
          dd_tau_tau,
          trial_wv);
    } else {
      ifdft_convert_response_vvars_uks(
          npoints,
          nfeatures,
          drho,
          dsigma,
          dtau,
          rho_gradient,
          drho_gradient,
          feature_gradient,
          d_sigma,
          dd_rho_rho,
          dd_rho_sigma,
          dd_rho_tau,
          dd_sigma_sigma,
          dd_sigma_tau,
          dd_tau_tau,
          trial_wv);
    }
  }

cleanup:
  ifdft_free_pyscf_mgga_input(workspace, rho, sigma, tau, rho_gradient);
  ifdft_free_pyscf_mgga_direction(workspace, drho, dsigma, dtau, drho_gradient);
  return rc;
}

static int
ifdft_parallel_eval_response_batch_from_pyscf_mgga(
    IFDFT_IFXC_Evaluator *evaluator,
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
    size_t errbuf_size)
{
  size_t nworkers;
  size_t nfeatures = 0;
  size_t output_components;
  size_t worker_index;
  int *statuses = NULL;
  int rc;
  ifxc_status status;

  if(evaluator == NULL || ntrial == 0 || weights == NULL
      || feature_gradient == NULL || weighted_kernel_vvars == NULL
      || dfeatures == NULL){
    ifdft_set_error(errbuf, errbuf_size,
                    "missing parallel PySCF MGGA response batch input");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  rc = ifdft_validate_pyscf_mgga_input(
      spin, npoints, rho0_a, rho0_b, ncomponents, errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  nworkers = ifdft_parallel_worker_count(evaluator, npoints);
  rc = ifdft_evaluator_require_workers(evaluator, nworkers, errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  {
    ifxc_func_type *func = NULL;
    rc = ifdft_evaluator_get_worker_func(
        evaluator, 0, spin, &func, errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      return rc;
    }
    status = ifxc_nfeatures(func, &nfeatures);
    if(status != IFXC_OK){
      return ifdft_status(status, errbuf, errbuf_size);
    }
    if(nworkers == 1){
      return ifdft_eval_response_batch_from_pyscf_mgga_with_func(
          func, &evaluator->workers[0].workspace, spin,
          npoints, npoints, 0, ntrial, rho0_a, rho0_b,
          rho1_a_batch, rho1_b_batch, ncomponents, weights, feature_gradient,
          weighted_kernel_vvars, dfeatures, errbuf, errbuf_size);
    }
  }

  output_components = (spin == 0) ? 5 : 10;
  statuses = (int *)calloc(nworkers, sizeof(int));
  if(statuses == NULL){
    ifdft_set_error(errbuf, errbuf_size, "failed to allocate worker statuses");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  for(worker_index = 0; worker_index < nworkers; ++worker_index){
    ifxc_func_type *func = NULL;
    size_t start = npoints * worker_index / nworkers;
    size_t end = npoints * (worker_index + 1) / nworkers;
    size_t chunk_size = end - start;
    rc = ifdft_evaluator_get_worker_func(
        evaluator, worker_index, spin, &func, errbuf, errbuf_size);
    if(rc == IFDFT_IFXC_OK){
      rc = ifdft_workspace_require(
          &evaluator->workers[worker_index].workspace.weighted_vvars,
          &evaluator->workers[worker_index].workspace.weighted_vvars_capacity,
          ntrial * output_components * chunk_size,
          "worker response batch", errbuf, errbuf_size);
    }
    if(rc == IFDFT_IFXC_OK){
      rc = ifdft_workspace_require(
          &evaluator->workers[worker_index].workspace.features,
          &evaluator->workers[worker_index].workspace.features_capacity,
          ntrial * nfeatures,
          "worker response feature derivatives", errbuf, errbuf_size);
    }
    if(rc != IFDFT_IFXC_OK){
      free(statuses);
      return rc;
    }
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1) num_threads((int)nworkers)
#endif
  for(worker_index = 0; worker_index < nworkers; ++worker_index){
    IFDFT_IFXC_Worker *worker = &evaluator->workers[worker_index];
    size_t start = npoints * worker_index / nworkers;
    size_t end = npoints * (worker_index + 1) / nworkers;
    size_t chunk_size = end - start;
    statuses[worker_index] = ifdft_eval_response_batch_from_pyscf_mgga_with_func(
        &worker->funcs[ifdft_spin_index(spin)], &worker->workspace, spin,
        chunk_size, npoints, start, ntrial, rho0_a, rho0_b,
        rho1_a_batch, rho1_b_batch, ncomponents, weights + start,
        feature_gradient, worker->workspace.weighted_vvars,
        worker->workspace.features, NULL, 0);
  }

  memset(dfeatures, 0, ntrial * nfeatures * sizeof(double));
  rc = IFDFT_IFXC_OK;
  for(worker_index = 0; worker_index < nworkers; ++worker_index){
    size_t start = npoints * worker_index / nworkers;
    size_t end = npoints * (worker_index + 1) / nworkers;
    size_t chunk_size = end - start;
    size_t trial;
    if(statuses[worker_index] != IFDFT_IFXC_OK){
      rc = statuses[worker_index];
      ifdft_set_error(errbuf, errbuf_size,
                      "parallel PySCF MGGA response batch worker failed");
      break;
    }
    for(trial = 0; trial < ntrial; ++trial){
      size_t component;
      size_t feature;
      for(component = 0; component < output_components; ++component){
        memcpy(
            weighted_kernel_vvars
                + (trial * output_components + component) * npoints + start,
            evaluator->workers[worker_index].workspace.weighted_vvars
                + (trial * output_components + component) * chunk_size,
            chunk_size * sizeof(double));
      }
      for(feature = 0; feature < nfeatures; ++feature){
        dfeatures[trial * nfeatures + feature] +=
            evaluator->workers[worker_index].workspace.features[
                trial * nfeatures + feature];
      }
    }
  }
  free(statuses);
  return rc;
}

static int
ifdft_eval_weighted_response_vvars_with_func(
    const ifxc_func_type *func,
    IFDFT_Workspace *workspace,
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
    size_t errbuf_size)
{
  ifxc_dimensions_t dims;
  size_t nfeatures = 0;
  double *d_rho = NULL;
  double *d_sigma = NULL;
  double *d_tau = NULL;
  double *dd_rho_rho = NULL;
  double *dd_rho_sigma = NULL;
  double *dd_rho_tau = NULL;
  double *dd_sigma_sigma = NULL;
  double *dd_sigma_tau = NULL;
  double *dd_tau_tau = NULL;
  int rc;
  ifxc_status status;

  if(func == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing evaluator function");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  status = ifxc_func_dimensions(func, &dims);
  if(status != IFXC_OK){
    return ifdft_status(status, errbuf, errbuf_size);
  }
  status = ifxc_nfeatures(func, &nfeatures);
  if(status != IFXC_OK){
    return ifdft_status(status, errbuf, errbuf_size);
  }
  rc = ifdft_validate_response_vvars_args(
      &dims,
      spin,
      npoints,
      rho,
      sigma,
      tau,
      drho,
      dsigma,
      dtau,
      rho_gradient,
      drho_gradient,
      weights,
      feature_gradient,
      weighted_vvars,
      errbuf,
      errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }

  if(workspace == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing evaluator workspace");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  rc = ifdft_workspace_require(&workspace->feature_d_rho, &workspace->feature_d_rho_capacity,
      npoints * nfeatures * dims.rho, "feature d_rho", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  rc = ifdft_workspace_require(&workspace->feature_d_sigma, &workspace->feature_d_sigma_capacity,
      npoints * nfeatures * dims.sigma, "feature d_sigma", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  rc = ifdft_workspace_require(&workspace->feature_d_tau, &workspace->feature_d_tau_capacity,
      npoints * nfeatures * dims.tau, "feature d_tau", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  rc = ifdft_workspace_require(&workspace->dd_rho_rho, &workspace->dd_rho_rho_capacity,
      npoints * nfeatures * ifdft_sym_count(dims.rho), "dd_rho_rho", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  rc = ifdft_workspace_require(&workspace->dd_rho_sigma, &workspace->dd_rho_sigma_capacity,
      npoints * nfeatures * dims.rho * dims.sigma, "dd_rho_sigma", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  rc = ifdft_workspace_require(&workspace->dd_rho_tau, &workspace->dd_rho_tau_capacity,
      npoints * nfeatures * dims.rho * dims.tau, "dd_rho_tau", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  rc = ifdft_workspace_require(&workspace->dd_sigma_sigma, &workspace->dd_sigma_sigma_capacity,
      npoints * nfeatures * ifdft_sym_count(dims.sigma), "dd_sigma_sigma", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  rc = ifdft_workspace_require(&workspace->dd_sigma_tau, &workspace->dd_sigma_tau_capacity,
      npoints * nfeatures * dims.sigma * dims.tau, "dd_sigma_tau", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  rc = ifdft_workspace_require(&workspace->dd_tau_tau, &workspace->dd_tau_tau_capacity,
      npoints * nfeatures * ifdft_sym_count(dims.tau), "dd_tau_tau", errbuf, errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  d_rho = workspace->feature_d_rho;
  d_sigma = workspace->feature_d_sigma;
  d_tau = workspace->feature_d_tau;
  dd_rho_rho = workspace->dd_rho_rho;
  dd_rho_sigma = workspace->dd_rho_sigma;
  dd_rho_tau = workspace->dd_rho_tau;
  dd_sigma_sigma = workspace->dd_sigma_sigma;
  dd_sigma_tau = workspace->dd_sigma_tau;
  dd_tau_tau = workspace->dd_tau_tau;

  rc = ifdft_eval_block_with_func(
      func,
      spin,
      npoints,
      rho,
      sigma,
      tau,
      weights,
      IFDFT_IFXC_WANT_FIRST_DERIVATIVES | IFDFT_IFXC_WANT_SECOND_DERIVATIVES,
      NULL,
      NULL,
      d_rho,
      d_sigma,
      d_tau,
      dd_rho_rho,
      dd_rho_sigma,
      dd_rho_tau,
      dd_sigma_sigma,
      dd_sigma_tau,
      dd_tau_tau,
      errbuf,
      errbuf_size);
  if(rc == IFDFT_IFXC_OK){
    if(spin == 0){
      ifdft_convert_response_vvars_rks(
          npoints,
          nfeatures,
          drho,
          dsigma,
          dtau,
          rho_gradient,
          drho_gradient,
          feature_gradient,
          d_sigma,
          dd_rho_rho,
          dd_rho_sigma,
          dd_rho_tau,
          dd_sigma_sigma,
          dd_sigma_tau,
          dd_tau_tau,
          weighted_vvars);
    } else {
      ifdft_convert_response_vvars_uks(
          npoints,
          nfeatures,
          drho,
          dsigma,
          dtau,
          rho_gradient,
          drho_gradient,
          feature_gradient,
          d_sigma,
          dd_rho_rho,
          dd_rho_sigma,
          dd_rho_tau,
          dd_sigma_sigma,
          dd_sigma_tau,
          dd_tau_tau,
          weighted_vvars);
    }
  }

  return rc;
}

int
IFDFT_ifxc_version(int *major, int *minor, int *patch)
{
  if(major == NULL || minor == NULL || patch == NULL){
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  *major = ifxc_version_major();
  *minor = ifxc_version_minor();
  *patch = ifxc_version_patch();
  return IFDFT_IFXC_OK;
}

int
IFDFT_ifxc_feature_set_count(size_t *count)
{
  if(count == NULL){
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  *count = IFDFT_FEATURE_SET_COUNT;
  return IFDFT_IFXC_OK;
}

int
IFDFT_ifxc_feature_set_key(size_t set_index, const char **key)
{
  if(key == NULL){
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  *key = NULL;
  if(set_index >= IFDFT_FEATURE_SET_COUNT){
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  *key = IFDFT_FEATURE_SETS[set_index].key;
  return IFDFT_IFXC_OK;
}

int
IFDFT_ifxc_feature_count(const char *feature_set_key, size_t *count)
{
  const ifdft_feature_set_entry *entry = NULL;
  const ifxc_feature_set_info_t *info = NULL;
  ifxc_status status;
  int rc;

  if(count == NULL){
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  rc = ifdft_resolve_feature_set_key(feature_set_key, &entry);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  status = ifxc_feature_set_info(entry->feature_set, &info);
  if(status != IFXC_OK){
    return IFDFT_IFXC_E_IFXC_STATUS;
  }
  *count = info->nfeatures;
  return IFDFT_IFXC_OK;
}

int
IFDFT_ifxc_feature_key(const char *feature_set_key, size_t feature_index, const char **key)
{
  const ifdft_feature_set_entry *entry = NULL;
  const ifxc_feature_info_t *info = NULL;
  ifxc_status status;
  int rc;

  if(key == NULL){
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  *key = NULL;
  rc = ifdft_resolve_feature_set_key(feature_set_key, &entry);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  status = ifxc_feature_info(entry->feature_set, feature_index, &info);
  if(status != IFXC_OK){
    return IFDFT_IFXC_E_IFXC_STATUS;
  }
  *key = info->key;
  return IFDFT_IFXC_OK;
}

int
IFDFT_ifxc_feature_set_max_deriv_order(const char *feature_set_key, int *max_deriv_order)
{
  const ifdft_feature_set_entry *entry = NULL;
  const ifxc_feature_set_info_t *info = NULL;
  ifxc_status status;
  int rc;

  if(max_deriv_order == NULL){
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  rc = ifdft_resolve_feature_set_key(feature_set_key, &entry);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  status = ifxc_feature_set_info(entry->feature_set, &info);
  if(status != IFXC_OK){
    return IFDFT_IFXC_E_IFXC_STATUS;
  }
  *max_deriv_order = (int)info->max_deriv_order;
  return IFDFT_IFXC_OK;
}

int
IFDFT_ifxc_feature_set_xctype(const char *feature_set_key, const char **xctype)
{
  const ifdft_feature_set_entry *entry = NULL;
  int rc;

  if(xctype == NULL){
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  *xctype = NULL;
  rc = ifdft_resolve_feature_set_key(feature_set_key, &entry);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }
  *xctype = entry->xctype;
  return IFDFT_IFXC_OK;
}

static int
ifdft_validate_eval_args(
    const ifxc_dimensions_t *dims,
    size_t npoints,
    const double *rho,
    const double *sigma,
    const double *tau,
    const double *weights,
    unsigned int flags,
    const double *features_integral,
    const double *local_values,
    const double *d_rho,
    const double *d_sigma,
    const double *d_tau,
    const double *dd_rho_rho,
    const double *dd_rho_sigma,
    const double *dd_rho_tau,
    const double *dd_sigma_sigma,
    const double *dd_sigma_tau,
    const double *dd_tau_tau,
    char *errbuf,
    size_t errbuf_size)
{
  unsigned int weighted_flags;

  if(dims == NULL || npoints == 0 || rho == NULL || sigma == NULL || tau == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing integral-feature block input");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  if(dims->rho == 0 || dims->sigma == 0 || dims->tau == 0){
    ifdft_set_error(errbuf, errbuf_size, "unsupported integral-feature dimensions");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }

  weighted_flags = IFDFT_IFXC_WANT_INTEGRAL_FEATURES
      | IFDFT_IFXC_WANT_FIRST_DERIVATIVES
      | IFDFT_IFXC_WANT_SECOND_DERIVATIVES;
  if((flags & weighted_flags) != 0 && weights == NULL){
    ifdft_set_error(errbuf, errbuf_size, "weights are required for integral feature outputs");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  if((flags & IFDFT_IFXC_WANT_INTEGRAL_FEATURES) != 0 && features_integral == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing integral feature output buffer");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  if((flags & IFDFT_IFXC_WANT_LOCAL_VALUES) != 0 && local_values == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing local value output buffer");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  if((flags & IFDFT_IFXC_WANT_FIRST_DERIVATIVES) != 0
      && (d_rho == NULL || d_sigma == NULL || d_tau == NULL)){
    ifdft_set_error(errbuf, errbuf_size, "missing first derivative output buffer");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  if((flags & IFDFT_IFXC_WANT_SECOND_DERIVATIVES) != 0
      && (dd_rho_rho == NULL || dd_rho_sigma == NULL || dd_rho_tau == NULL
          || dd_sigma_sigma == NULL || dd_sigma_tau == NULL || dd_tau_tau == NULL)){
    ifdft_set_error(errbuf, errbuf_size, "missing second derivative output buffer");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }
  return IFDFT_IFXC_OK;
}

static int
ifdft_add_entry(
    const ifxc_func_type *func,
    const ifxc_input *input,
    ifxc_deriv_entry *entries,
    size_t *n_entries,
    ifxc_target target,
    unsigned int order,
    const ifxc_variable *vars,
    double *out,
    char *errbuf,
    size_t errbuf_size)
{
  size_t n_double = 0;
  ifxc_status status;

  entries[*n_entries].target = target;
  entries[*n_entries].order = order;
  entries[*n_entries].vars = vars;
  entries[*n_entries].out = out;

  status = ifxc_output_size(func, input, &entries[*n_entries], &n_double);
  if(status != IFXC_OK){
    return ifdft_status(status, errbuf, errbuf_size);
  }
  (void)n_double;
  *n_entries += 1;
  return IFDFT_IFXC_OK;
}

static int
ifdft_eval_block_with_func(
    const ifxc_func_type *func,
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
    size_t errbuf_size)
{
  static const ifxc_variable rho_vars[1] = {IFXC_VAR_RHO};
  static const ifxc_variable sigma_vars[1] = {IFXC_VAR_SIGMA};
  static const ifxc_variable tau_vars[1] = {IFXC_VAR_TAU};
  static const ifxc_variable rho_rho_vars[2] = {IFXC_VAR_RHO, IFXC_VAR_RHO};
  static const ifxc_variable rho_sigma_vars[2] = {IFXC_VAR_RHO, IFXC_VAR_SIGMA};
  static const ifxc_variable rho_tau_vars[2] = {IFXC_VAR_RHO, IFXC_VAR_TAU};
  static const ifxc_variable sigma_sigma_vars[2] = {IFXC_VAR_SIGMA, IFXC_VAR_SIGMA};
  static const ifxc_variable sigma_tau_vars[2] = {IFXC_VAR_SIGMA, IFXC_VAR_TAU};
  static const ifxc_variable tau_tau_vars[2] = {IFXC_VAR_TAU, IFXC_VAR_TAU};

  ifxc_input input;
  ifxc_deriv_entry entries[11];
  ifxc_dimensions_t dims;
  size_t n_entries = 0;
  int rc;
  ifxc_status status;

  (void)spin;

  if(func == NULL){
    ifdft_set_error(errbuf, errbuf_size, "missing evaluator function");
    return IFDFT_IFXC_E_INVALID_ARGUMENT;
  }

  status = ifxc_func_dimensions(func, &dims);
  if(status != IFXC_OK){
    return ifdft_status(status, errbuf, errbuf_size);
  }

  rc = ifdft_validate_eval_args(
      &dims,
      npoints,
      rho,
      sigma,
      tau,
      weights,
      flags,
      features_integral,
      local_values,
      d_rho,
      d_sigma,
      d_tau,
      dd_rho_rho,
      dd_rho_sigma,
      dd_rho_tau,
      dd_sigma_sigma,
      dd_sigma_tau,
      dd_tau_tau,
      errbuf,
      errbuf_size);
  if(rc != IFDFT_IFXC_OK){
    return rc;
  }

  input.npoints = npoints;
  input.rho = rho;
  input.sigma = sigma;
  input.lapl = NULL;
  input.tau = tau;
  input.weights = weights;

  if((flags & IFDFT_IFXC_WANT_INTEGRAL_FEATURES) != 0){
    rc = ifdft_add_entry(func, &input, entries, &n_entries,
        IFXC_TARGET_INTEGRAL, 0, NULL, features_integral, errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      return rc;
    }
  }
  if((flags & IFDFT_IFXC_WANT_LOCAL_VALUES) != 0){
    rc = ifdft_add_entry(func, &input, entries, &n_entries,
        IFXC_TARGET_LOCAL, 0, NULL, local_values, errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      return rc;
    }
  }
  if((flags & IFDFT_IFXC_WANT_FIRST_DERIVATIVES) != 0){
    rc = ifdft_add_entry(func, &input, entries, &n_entries,
        IFXC_TARGET_INTEGRAL, 1, rho_vars, d_rho, errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      return rc;
    }
    rc = ifdft_add_entry(func, &input, entries, &n_entries,
        IFXC_TARGET_INTEGRAL, 1, sigma_vars, d_sigma, errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      return rc;
    }
    rc = ifdft_add_entry(func, &input, entries, &n_entries,
        IFXC_TARGET_INTEGRAL, 1, tau_vars, d_tau, errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      return rc;
    }
  }
  if((flags & IFDFT_IFXC_WANT_SECOND_DERIVATIVES) != 0){
    rc = ifdft_add_entry(func, &input, entries, &n_entries,
        IFXC_TARGET_INTEGRAL, 2, rho_rho_vars, dd_rho_rho, errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      return rc;
    }
    rc = ifdft_add_entry(func, &input, entries, &n_entries,
        IFXC_TARGET_INTEGRAL, 2, rho_sigma_vars, dd_rho_sigma, errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      return rc;
    }
    rc = ifdft_add_entry(func, &input, entries, &n_entries,
        IFXC_TARGET_INTEGRAL, 2, rho_tau_vars, dd_rho_tau, errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      return rc;
    }
    rc = ifdft_add_entry(func, &input, entries, &n_entries,
        IFXC_TARGET_INTEGRAL, 2, sigma_sigma_vars, dd_sigma_sigma, errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      return rc;
    }
    rc = ifdft_add_entry(func, &input, entries, &n_entries,
        IFXC_TARGET_INTEGRAL, 2, sigma_tau_vars, dd_sigma_tau, errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      return rc;
    }
    rc = ifdft_add_entry(func, &input, entries, &n_entries,
        IFXC_TARGET_INTEGRAL, 2, tau_tau_vars, dd_tau_tau, errbuf, errbuf_size);
    if(rc != IFDFT_IFXC_OK){
      return rc;
    }
  }

  status = ifxc_eval(func, &input, n_entries, entries);
  if(status != IFXC_OK){
    return ifdft_status(status, errbuf, errbuf_size);
  }
  return IFDFT_IFXC_OK;
}
