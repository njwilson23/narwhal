import ctypes
from ctypes import c_double

cgsw = ctypes.cdll.LoadLibrary("gsw.cpython-34m.so")


cgsw.adiabatic_lapse_rate_from_ct.argtypes = (c_double, double)
cgsw.alpha.argtypes = (c_double, double)
cgsw.alpha_on_beta.argtypes = (c_double, double)
cgsw.alpha_wrt_t_exact.argtypes = (c_double, double)
cgsw.beta_const_t_exact.argtypes = (c_double, double)
cgsw.beta.argtypes = (c_double, double)
cgsw.c_from_sp.argtypes = (c_double, double)
cgsw.cabbeling.argtypes = (c_double, double)
cgsw.ct_freezing.argtypes = (c_double, double)
cgsw.ct_from_pt.argtypes = (c_double, c_double)
cgsw.ct_from_t.argtypes = (c_double, c_double, c_double)


cgsw.adiabatic_lapse_rate_from_ct.restype = c_double
cgsw.alpha.restype = c_double
cgsw.alpha_on_beta.restype = c_double
cgsw.alpha_wrt_t_exact.restype = c_double
cgsw.beta_const_t_exact.restype = c_double
cgsw.beta.restype = c_double
cgsw.c_from_sp.restype = c_double
cgsw.cabbeling.restype = c_double
cgsw.ct_freezing.restype = c_double
cgsw.ct_from_pt.restype = c_double
cgsw.ct_from_t.restype = c_double


adiabatic_lapse_rate_from_ct = cgsw.adiabatic_lapse_rate_from_ct
alpha                        = cgsw.alpha
alpha_on_beta                = cgsw.alpha_on_beta
alpha_wrt_t_exact            = cgsw.alpha_wrt_t_exact
beta_const_t_exact           = cgsw.beta_const_t_exact
beta                         = cgsw.beta
c_from_sp                    = cgsw.c_from_sp
cabbeling                    = cgsw.cabbeling
ct_freezing                  = cgsw.ct_freezing
ct_from_pt                   = cgsw.ct_from_pt
ct_from_t                    = cgsw.ct_from_t


# extern double gsw_deltasa_atlas(double p, double lon, double lat);
# extern double gsw_deltasa_from_sp(double sp, double p, double lon, double lat);
# extern double gsw_dynamic_enthalpy(double sa, double ct, double p);
# extern double gsw_enthalpy(double sa, double ct, double p);
# extern double gsw_enthalpy_sso_0_p(double p);
# extern double gsw_enthalpy_t_exact(double sa, double t, double p);
# extern double gsw_entropy_from_t(double sa, double t, double p);
# extern double gsw_entropy_part(double sa, double t, double p);
# extern double gsw_entropy_part_zerop(double sa, double pt0);
# extern double gsw_fdelta(double p, double lon, double lat);
# extern double gsw_gibbs(int ns, int nt, int np, double sa, double t, double p);
# extern double gsw_gibbs_pt0_pt0(double sa, double pt0);
# extern double gsw_grav(double lat, double p);
# extern double gsw_hill_ratio_at_sp2(double t);
# extern int    gsw_indx(double *x, int n, double z);
# extern double gsw_internal_energy(double sa, double ct, double p);
# extern void   gsw_ipv_vs_fnsquared_ratio(double *sa, double *ct, double *p,
# 			int nz, double *ipv_vs_fnsquared_ratio, double *p_mid);
# extern double gsw_kappa(double sa, double ct, double p);
# extern double gsw_kappa_t_exact(double sa, double t, double p);
# extern double gsw_latentheat_evap_ct(double sa, double ct);
# extern double gsw_latentheat_evap_t(double sa, double t);
# extern double gsw_latentheat_melting(double sa, double p);
# extern void gsw_nsquared(double *sa, double *ct, double *p, double *lat,
# 			int nz, double *n2, double *p_mid);
# extern double gsw_pot_rho_t_exact(double sa, double t, double p, double p_ref);
# extern double gsw_pt0_from_t(double sa, double t, double p);
# extern double gsw_pt_from_ct(double sa, double ct);
# extern double gsw_pt_from_t(double sa, double t, double p, double p_ref);
# extern double gsw_rho(double sa, double ct, double p);
# extern void   gsw_rho_first_derivatives(double sa, double ct, double p,
# 			double *drho_dsa, double *drho_dct, double *drho_dp);
# extern double gsw_rho_t_exact(double sa, double t, double p);
# extern double gsw_saar(double p, double lon, double lat);
# extern double gsw_sa_from_rho(double rho, double ct, double p);
# extern double gsw_sa_from_sp_baltic(double sp, double lon, double lat);
# extern double gsw_sa_from_sp(double sp, double p, double lon, double lat);
# extern double gsw_sa_from_sstar(double sstar, double p,double lon,double lat);
# extern double gsw_sigma0(double sa, double ct);
# extern double gsw_sigma1(double sa, double ct);
# extern double gsw_sigma2(double sa, double ct);
# extern double gsw_sigma3(double sa, double ct);
# extern double gsw_sigma4(double sa, double ct);
# extern double gsw_sound_speed(double sa, double ct, double p);
# extern double gsw_sound_speed_t_exact(double sa, double t, double p);
# extern double gsw_specvol_anom(double sa, double ct, double p);
# extern double gsw_specvol(double sa, double ct, double p);
# extern double gsw_specvol_sso_0_p(double p);
# extern double gsw_specvol_t_exact(double sa, double t, double p);
# extern double gsw_sp_from_c(double c, double t, double p);
# extern double gsw_sp_from_sa_baltic(double sa, double lon, double lat);
# extern double gsw_sp_from_sa(double sa, double p, double lon, double lat);
# extern double gsw_sp_from_sk(double sk);
# extern double gsw_sp_from_sr(double sr);
# extern double gsw_sp_from_sstar(double sstar, double p,double lon,double lat);
# extern double gsw_sr_from_sp(double sp);
# extern double gsw_sstar_from_sa(double sa, double p, double lon, double lat);
# extern double gsw_sstar_from_sp(double sp, double p, double lon, double lat);
# extern double gsw_t_freezing(double sa, double p, double saturation_fraction);
# extern double gsw_t_from_ct(double sa, double ct, double p);
# extern double gsw_thermobaric(double sa, double ct, double p);
# extern void   gsw_turner_rsubrho(double *sa, double *ct, double *p,
# 			int nz, double *tu, double *rsubrho, double *p_mid);
# extern double gsw_xinterp1(double *x, double *y, int n, double x0);
# extern double gsw_z_from_p(double p, double lat);


