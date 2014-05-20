""" Dynamically builds Python wrappers for Gibbs Seawater Toolbox at runtime """

import ctypes
from os import listdir
from os.path import dirname, realpath, splitext, join

install_dir = dirname(realpath(__file__))

def find_gsw(s):
    return splitext(s)[1] == ".so" and s.startswith("cgsw")

name = list(filter(find_gsw, listdir(install_dir)))[0]
cgsw = ctypes.cdll.LoadLibrary(join(install_dir, name))

header = \
"""
extern void   gsw_add_barrier(double *input_data, double lon, double lat,
              double long_grid, double lat_grid, double dlong_grid,
              double dlat_grid, double *output_data);
extern void   gsw_add_mean(double *data_in, double lon, double lat,
              double *data_out);
extern double gsw_adiabatic_lapse_rate_from_ct(double sa, double ct, double p);
extern double gsw_alpha(double sa, double ct, double p);
extern double gsw_alpha_on_beta(double sa, double ct, double p);
extern double gsw_alpha_wrt_t_exact(double sa, double t, double p);
extern double gsw_beta_const_t_exact(double sa, double t, double p);
extern double gsw_beta(double sa, double ct, double p);
extern double gsw_c_from_sp(double sp, double t, double p);
extern double gsw_cabbeling(double sa, double ct, double p);
extern double gsw_ct_freezing(double sa, double p, double saturation_fraction);
extern double gsw_ct_from_pt(double sa, double pt);
extern double gsw_ct_from_t(double sa, double t, double p);
extern double gsw_deltasa_atlas(double p, double lon, double lat);
extern double gsw_deltasa_from_sp(double sp, double p, double lon, double lat);
extern double gsw_dynamic_enthalpy(double sa, double ct, double p);
extern double gsw_enthalpy(double sa, double ct, double p);
extern double gsw_enthalpy_sso_0_p(double p);
extern double gsw_enthalpy_t_exact(double sa, double t, double p);
extern double gsw_entropy_from_t(double sa, double t, double p);
extern double gsw_entropy_part(double sa, double t, double p);
extern double gsw_entropy_part_zerop(double sa, double pt0);
extern double gsw_fdelta(double p, double lon, double lat);
extern double gsw_gibbs(int ns, int nt, int np, double sa, double t, double p);
extern double gsw_gibbs_pt0_pt0(double sa, double pt0);
extern double gsw_grav(double lat, double p);
extern double gsw_hill_ratio_at_sp2(double t);
extern int    gsw_indx(double *x, int n, double z);
extern double gsw_internal_energy(double sa, double ct, double p);
extern void   gsw_ipv_vs_fnsquared_ratio(double *sa, double *ct, double *p,
              int nz, double *ipv_vs_fnsquared_ratio, double *p_mid);
extern double gsw_kappa(double sa, double ct, double p);
extern double gsw_kappa_t_exact(double sa, double t, double p);
extern double gsw_latentheat_evap_ct(double sa, double ct);
extern double gsw_latentheat_evap_t(double sa, double t);
extern double gsw_latentheat_melting(double sa, double p);
extern void gsw_nsquared(double *sa, double *ct, double *p, double *lat,
              int nz, double *n2, double *p_mid);
extern double gsw_pot_rho_t_exact(double sa, double t, double p, double p_ref);
extern double gsw_pt0_from_t(double sa, double t, double p);
extern double gsw_pt_from_ct(double sa, double ct);
extern double gsw_pt_from_t(double sa, double t, double p, double p_ref);
extern double gsw_rho(double sa, double ct, double p);
extern void   gsw_rho_first_derivatives(double sa, double ct, double p,
              double *drho_dsa, double *drho_dct, double *drho_dp);
extern double gsw_rho_t_exact(double sa, double t, double p);
extern double gsw_saar(double p, double lon, double lat);
extern double gsw_sa_from_rho(double rho, double ct, double p);
extern double gsw_sa_from_sp_baltic(double sp, double lon, double lat);
extern double gsw_sa_from_sp(double sp, double p, double lon, double lat);
extern double gsw_sa_from_sstar(double sstar, double p,double lon,double lat);
extern double gsw_sigma0(double sa, double ct);
extern double gsw_sigma1(double sa, double ct);
extern double gsw_sigma2(double sa, double ct);
extern double gsw_sigma3(double sa, double ct);
extern double gsw_sigma4(double sa, double ct);
extern double gsw_sound_speed(double sa, double ct, double p);
extern double gsw_sound_speed_t_exact(double sa, double t, double p);
extern double gsw_specvol_anom(double sa, double ct, double p);
extern double gsw_specvol(double sa, double ct, double p);
extern double gsw_specvol_sso_0_p(double p);
extern double gsw_specvol_t_exact(double sa, double t, double p);
extern double gsw_sp_from_c(double c, double t, double p);
extern double gsw_sp_from_sa_baltic(double sa, double lon, double lat);
extern double gsw_sp_from_sa(double sa, double p, double lon, double lat);
extern double gsw_sp_from_sk(double sk);
extern double gsw_sp_from_sr(double sr);
extern double gsw_sp_from_sstar(double sstar, double p,double lon,double lat);
extern double gsw_sr_from_sp(double sp);
extern double gsw_sstar_from_sa(double sa, double p, double lon, double lat);
extern double gsw_sstar_from_sp(double sp, double p, double lon, double lat);
extern double gsw_t_freezing(double sa, double p, double saturation_fraction);
extern double gsw_t_from_ct(double sa, double ct, double p);
extern double gsw_thermobaric(double sa, double ct, double p);
extern void   gsw_turner_rsubrho(double *sa, double *ct, double *p,
              int nz, double *tu, double *rsubrho, double *p_mid);
extern double gsw_xinterp1(double *x, double *y, int n, double x0);
extern double gsw_z_from_p(double p, double lat);
"""

importnames = ["gsw_adiabatic_lapse_rate_from_ct",
               "gsw_alpha",
               "gsw_alpha_on_beta",
               "gsw_alpha_wrt_t_exact",
               "gsw_beta_const_t_exact",
               "gsw_beta",
               "gsw_c_from_sp",
               "gsw_cabbeling",
               "gsw_ct_freezing",
               "gsw_ct_from_pt",
               "gsw_ct_from_t",
               "gsw_deltasa_atlas",
               "gsw_deltasa_from_sp",
               "gsw_dynamic_enthalpy",
               "gsw_enthalpy",
               "gsw_enthalpy_sso_0_p",
               "gsw_enthalpy_t_exact",
               "gsw_entropy_from_t",
               "gsw_entropy_part",
               "gsw_entropy_part_zerop",
               "gsw_fdelta",
               "gsw_gibbs",
               "gsw_gibbs_pt0_pt0",
               "gsw_grav",
               "gsw_hill_ratio_at_sp2",
               "gsw_internal_energy",
               "gsw_kappa",
               "gsw_kappa_t_exact",
               "gsw_latentheat_evap_ct",
               "gsw_latentheat_evap_t",
               "gsw_latentheat_melting",
               "gsw_pot_rho_t_exact",
               "gsw_pt0_from_t",
               "gsw_pt_from_ct",
               "gsw_pt_from_t",
               "gsw_rho",
               "gsw_rho_t_exact",
               "gsw_saar",
               "gsw_sa_from_rho",
               "gsw_sa_from_sp_baltic",
               "gsw_sa_from_sp",
               "gsw_sa_from_sstar",
               "gsw_sigma0",
               "gsw_sigma1",
               "gsw_sigma2",
               "gsw_sigma3",
               "gsw_sigma4",
               "gsw_sound_speed",
               "gsw_sound_speed_t_exact",
               "gsw_specvol_anom",
               "gsw_specvol",
               "gsw_specvol_sso_0_p",
               "gsw_specvol_t_exact",
               "gsw_sp_from_c",
               "gsw_sp_from_sa_baltic",
               "gsw_sp_from_sa",
               "gsw_sp_from_sk",
               "gsw_sp_from_sr",
               "gsw_sp_from_sstar",
               "gsw_sr_from_sp",
               "gsw_sstar_from_sa",
               "gsw_sstar_from_sp",
               "gsw_t_freezing",
               "gsw_t_from_ct",
               "gsw_thermobaric",
               "gsw_z_from_p"]

lines = header.split("\n")
lines = filter(lambda s: s.startswith("extern double") and s.endswith(";"), lines)

def cname(line):
    return line.split(" ", 2)[2].split("(", 1)[0]

def getfuncpointer(name):
    return cgsw.__getattr__(name)

def argtypes(line):
    args = line.rsplit("(", 1)[1].split(")", 1)[0].split(",")
    cargs = []
    for arg in args:
        typ = arg.split()[0].strip()
        if typ == "double":
            cargs.append(ctypes.c_double)
        elif typ == "int":
            cargs.append(ctypes.c_int)
    return tuple(cargs)

def argnames(line):
    args = line.rsplit("(", 1)[1].split(")", 1)[0].split(",")
    names = []
    for arg in args:
        names.append(arg.split()[1].strip())
    return tuple(names)

def restype(line):
    s = line.split(" ", 2)[1:2][0].strip()
    if s == "double":
        return ctypes.c_double

def addname(line):
    name = line.split(" ", 2)[2].split("(", 1)[0]
    if name[:4] == "gsw_":
        exec("{0} = cgsw.{1}".format(name[4:], name), addname.__globals__)
    return

for line in lines:
    name = cname(line)
    if name in importnames:
        func = getfuncpointer(name)
        func.argtypes = argtypes(line)
        func.restype = restype(line)
        func.__doc__ = name + str(argnames(line))
        addname(line)

