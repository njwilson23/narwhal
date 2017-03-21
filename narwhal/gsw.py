""" Dynamically builds Python wrappers for Gibbs Seawater Toolbox at runtime """

import ctypes
from os import listdir
from os.path import dirname, realpath, splitext, join
import itertools
from numpy import vectorize, isnan, nan

install_dir = dirname(realpath(__file__))

def find_gsw(s):
    return splitext(s)[1] == ".so" and s.startswith("cgsw")

name = list(filter(find_gsw, listdir(install_dir)))[0]
cgsw = ctypes.cdll.LoadLibrary(join(install_dir, name))

header = \
"""
extern double gsw_adiabatic_lapse_rate_from_ct(double sa, double ct, double p);
extern double gsw_adiabatic_lapse_rate_ice(double t, double p);
extern double gsw_alpha(double sa, double ct, double p);
extern double gsw_alpha_on_beta(double sa, double ct, double p);
extern double gsw_alpha_wrt_t_exact(double sa, double t, double p);
extern double gsw_alpha_wrt_t_ice(double t, double p);
extern double gsw_beta_const_t_exact(double sa, double t, double p);
extern double gsw_beta(double sa, double ct, double p);
extern double gsw_cabbeling(double sa, double ct, double p);
extern double gsw_c_from_sp(double sp, double t, double p);
extern double gsw_chem_potential_water_ice(double t, double p);
extern double gsw_chem_potential_water_t_exact(double sa, double t, double p);
extern double gsw_cp_ice(double t, double p);
extern double gsw_cp_t_exact(double sa, double t, double p);
extern double gsw_ct_freezing(double sa, double p, double saturation_fraction);
extern double gsw_ct_freezing_exact(double sa, double p, double saturation_fraction);
extern double gsw_ct_freezing_poly(double sa, double p, double saturation_fraction);
extern double gsw_ct_from_enthalpy(double sa, double h, double p);
extern double gsw_ct_from_enthalpy_exact(double sa, double h, double p);
extern double gsw_ct_from_entropy(double sa, double entropy);
extern double gsw_ct_from_pt(double sa, double pt);
extern double gsw_ct_from_t(double sa, double t, double p);
extern double gsw_ct_maxdensity(double sa, double p);
extern double gsw_deltasa_atlas(double p, double lon, double lat);
extern double gsw_deltasa_from_sp(double sp, double p, double lon, double lat);
extern double gsw_dilution_coefficient_t_exact(double sa, double t, double p);
extern double gsw_dynamic_enthalpy(double sa, double ct, double p);
extern double gsw_enthalpy_ct_exact(double sa, double ct, double p);
extern double gsw_enthalpy_diff(double sa, double ct, double p_shallow, double p_deep);
extern double gsw_enthalpy(double sa, double ct, double p);
extern double gsw_enthalpy_ice(double t, double p);
extern double gsw_enthalpy_sso_0(double p);
extern double gsw_enthalpy_t_exact(double sa, double t, double p);
extern double gsw_entropy_from_pt(double sa, double pt);
extern double gsw_entropy_from_t(double sa, double t, double p);
extern double gsw_entropy_ice(double t, double p);
extern double gsw_entropy_part(double sa, double t, double p);
extern double gsw_entropy_part_zerop(double sa, double pt0);
extern double gsw_fdelta(double p, double lon, double lat);
extern double *gsw_geo_strf_dyn_height(double *sa, double *ct, double *p, double p_ref, int n_levels, double *dyn_height);
extern double *gsw_geo_strf_dyn_height_pc(double *sa, double *ct, double *delta_p, int n_levels, double *geo_strf_dyn_height_pc, double *p_mid);
extern double gsw_gibbs_ice (int nt, int np, double t, double p);
extern double gsw_gibbs_ice_part_t(double t, double p);
extern double gsw_gibbs_ice_pt0(double pt0);
extern double gsw_gibbs_ice_pt0_pt0(double pt0);
extern double gsw_gibbs(int ns, int nt, int np, double sa, double t, double p);
extern double gsw_gibbs_pt0_pt0(double sa, double pt0);
extern double gsw_grav(double lat, double p);
extern double gsw_helmholtz_energy_ice(double t, double p);
extern double gsw_hill_ratio_at_sp2(double t);
extern double gsw_internal_energy(double sa, double ct, double p);
extern double gsw_internal_energy_ice(double t, double p);
extern double gsw_kappa_const_t_ice(double t, double p);
extern double gsw_kappa(double sa, double ct, double p);
extern double gsw_kappa_ice(double t, double p);
extern double gsw_kappa_t_exact(double sa, double t, double p);
extern double gsw_latentheat_evap_ct(double sa, double ct);
extern double gsw_latentheat_evap_t(double sa, double t);
extern double gsw_latentheat_melting(double sa, double p);
extern double gsw_melting_ice_equilibrium_sa_ct_ratio(double sa, double p);
extern double gsw_melting_ice_equilibrium_sa_ct_ratio_poly(double sa, double p);
extern double gsw_melting_ice_sa_ct_ratio(double sa, double ct, double p, double t_ih);
extern double gsw_melting_ice_sa_ct_ratio_poly(double sa, double ct, double p, double t_ih);
extern double gsw_melting_seaice_equilibrium_sa_ct_ratio(double sa, double p);
extern double gsw_melting_seaice_equilibrium_sa_ct_ratio_poly(double sa, double p);
extern double gsw_melting_seaice_sa_ct_ratio(double sa, double ct, double p, double sa_seaice, double t_seaice);
extern double gsw_melting_seaice_sa_ct_ratio_poly(double sa, double ct, double p, double sa_seaice, double t_seaice);
extern double gsw_pot_enthalpy_from_pt_ice(double pt0_ice);
extern double gsw_pot_enthalpy_from_pt_ice_poly(double pt0_ice);
extern double gsw_pot_enthalpy_ice_freezing(double sa, double p);
extern double gsw_pot_enthalpy_ice_freezing_poly(double sa, double p);
extern double gsw_pot_rho_t_exact(double sa, double t, double p, double p_ref);
extern double gsw_pressure_coefficient_ice(double t, double p);
extern double gsw_pressure_freezing_ct(double sa, double ct, double saturation_fraction);
extern double gsw_pt0_cold_ice_poly(double pot_enthalpy_ice);
extern double gsw_pt0_from_t(double sa, double t, double p);
extern double gsw_pt0_from_t_ice(double t, double p);
extern double gsw_pt_from_ct(double sa, double ct);
extern double gsw_pt_from_entropy(double sa, double entropy);
extern double gsw_pt_from_pot_enthalpy_ice(double pot_enthalpy_ice);
extern double gsw_pt_from_pot_enthalpy_ice_poly_dh(double pot_enthalpy_ice);
extern double gsw_pt_from_pot_enthalpy_ice_poly(double pot_enthalpy_ice);
extern double gsw_pt_from_t(double sa, double t, double p, double p_ref);
extern double gsw_pt_from_t_ice(double t, double p, double p_ref);
extern double gsw_rho(double sa, double ct, double p);
extern double gsw_rho_ice(double t, double p);
extern double gsw_rho_t_exact(double sa, double t, double p);
extern double gsw_saar(double p, double lon, double lat);
extern double gsw_sa_freezing_estimate(double p, double saturation_fraction, double *ct, double *t);
extern double gsw_sa_freezing_from_ct(double ct, double p, double saturation_fraction);
extern double gsw_sa_freezing_from_ct_poly(double ct, double p, double saturation_fraction);
extern double gsw_sa_freezing_from_t(double t, double p, double saturation_fraction);
extern double gsw_sa_freezing_from_t_poly(double t, double p, double saturation_fraction);
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
extern double gsw_sound_speed_ice(double t, double p);
extern double gsw_sound_speed_t_exact(double sa, double t, double p);
extern double gsw_specvol_anom_standard(double sa, double ct, double p);
extern double gsw_specvol(double sa, double ct, double p);
extern double gsw_specvol_ice(double t, double p);
extern double gsw_specvol_sso_0(double p);
extern double gsw_specvol_t_exact(double sa, double t, double p);
extern double gsw_sp_from_c(double c, double t, double p);
extern double gsw_sp_from_sa_baltic(double sa, double lon, double lat);
extern double gsw_sp_from_sa(double sa, double p, double lon, double lat);
extern double gsw_sp_from_sk(double sk);
extern double gsw_sp_from_sr(double sr);
extern double gsw_sp_from_sstar(double sstar, double p,double lon,double lat);
extern double gsw_spiciness0(double sa, double ct);
extern double gsw_spiciness1(double sa, double ct);
extern double gsw_spiciness2(double sa, double ct);
extern double gsw_sr_from_sp(double sp);
extern double gsw_sstar_from_sa(double sa, double p, double lon, double lat);
extern double gsw_sstar_from_sp(double sp, double p, double lon, double lat);
extern double gsw_t_deriv_chem_potential_water_t_exact(double sa, double t, double p);
extern double gsw_t_freezing(double sa, double p, double saturation_fraction);
extern double gsw_t_freezing_exact(double sa, double p, double saturation_fraction);
extern double gsw_t_freezing_poly(double sa, double p, double saturation_fraction, int polynomial);
extern double gsw_t_from_ct(double sa, double ct, double p);
extern double gsw_t_from_pt0_ice(double pt0_ice, double p);
extern double gsw_thermobaric(double sa, double ct, double p);
extern double *gsw_util_interp1q_int(int nx, double *x, int *iy, int nxi, double *x_i, double *y_i);
extern double gsw_util_xinterp1(double *x, double *y, int n, double x0);
extern double gsw_z_from_p(double p, double lat);
"""

importnames = ["gsw_adiabatic_lapse_rate_from_ct",
               "gsw_adiabatic_lapse_rate_ice",
               "gsw_alpha",
               "gsw_alpha_on_beta",
               "gsw_alpha_wrt_t_exact",
               "gsw_alpha_wrt_t_ice",
               "gsw_beta_const_t_exact",
               "gsw_beta",
               "gsw_cabbeling",
               "gsw_c_from_sp",
               "gsw_chem_potential_water_ice",
               "gsw_chem_potential_water_t_exact",
               "gsw_cp_ice",
               "gsw_cp_t_exact",
               "gsw_ct_freezing",
               "gsw_ct_freezing_exact",
               "gsw_ct_freezing_poly",
               "gsw_ct_from_enthalpy",
               "gsw_ct_from_enthalpy_exact",
               "gsw_ct_from_entropy",
               "gsw_ct_from_pt",
               "gsw_ct_from_t",
               "gsw_ct_maxdensity",
               "gsw_deltasa_atlas",
               "gsw_deltasa_from_sp",
               "gsw_dilution_coefficient_t_exact",
               "gsw_dynamic_enthalpy",
               "gsw_enthalpy_ct_exact",
               "gsw_enthalpy_diff",
               "gsw_enthalpy",
               "gsw_enthalpy_ice",
               "gsw_enthalpy_sso_0",
               "gsw_enthalpy_t_exact",
               "gsw_entropy_from_pt",
               "gsw_entropy_from_t",
               "gsw_entropy_ice",
               "gsw_entropy_part",
               "gsw_entropy_part_zerop",
               "gsw_fdelta",
               "gsw_gibbs_ice",
               "gsw_gibbs_ice_part_t",
               "gsw_gibbs_ice_pt0",
               "gsw_gibbs_ice_pt0_pt0",
               "gsw_gibbs",
               "gsw_gibbs_pt0_pt0",
               "gsw_grav",
               "gsw_helmholtz_energy_ice",
               "gsw_hill_ratio_at_sp2",
               "gsw_internal_energy",
               "gsw_internal_energy_ice",
               "gsw_kappa_const_t_ice",
               "gsw_kappa",
               "gsw_kappa_ice",
               "gsw_kappa_t_exact",
               "gsw_latentheat_evap_ct",
               "gsw_latentheat_evap_t",
               "gsw_latentheat_melting",
               "gsw_melting_ice_equilibrium_sa_ct_ratio",
               "gsw_melting_ice_equilibrium_sa_ct_ratio_poly",
               "gsw_melting_ice_sa_ct_ratio",
               "gsw_melting_ice_sa_ct_ratio_poly",
               "gsw_melting_seaice_equilibrium_sa_ct_ratio",
               "gsw_melting_seaice_equilibrium_sa_ct_ratio_poly",
               "gsw_melting_seaice_sa_ct_ratio",
               "gsw_melting_seaice_sa_ct_ratio_poly",
               "gsw_pot_enthalpy_from_pt_ice",
               "gsw_pot_enthalpy_from_pt_ice_poly",
               "gsw_pot_enthalpy_ice_freezing",
               "gsw_pot_enthalpy_ice_freezing_poly",
               "gsw_pot_rho_t_exact",
               "gsw_pressure_coefficient_ice",
               "gsw_pressure_freezing_ct",
               "gsw_pt0_cold_ice_poly",
               "gsw_pt0_from_t",
               "gsw_pt0_from_t_ice",
               "gsw_pt_from_ct",
               "gsw_pt_from_entropy",
               "gsw_pt_from_pot_enthalpy_ice",
               "gsw_pt_from_pot_enthalpy_ice_poly_dh",
               "gsw_pt_from_pot_enthalpy_ice_poly",
               "gsw_pt_from_t",
               "gsw_pt_from_t_ice",
               "gsw_rho",
               "gsw_rho_ice",
               "gsw_rho_t_exact",
               "gsw_saar",
               "gsw_sa_freezing_estimate",
               "gsw_sa_freezing_from_ct",
               "gsw_sa_freezing_from_ct_poly",
               "gsw_sa_freezing_from_t",
               "gsw_sa_freezing_from_t_poly",
               "gsw_sa_from_rho",
               "gsw_sa_from_sp_baltic",
               "gsw_sa_from_sp",
               "gsw_sa_from_sstar",
               "gsw_sa_p_inrange",
               "gsw_sigma0",
               "gsw_sigma1",
               "gsw_sigma2",
               "gsw_sigma3",
               "gsw_sigma4",
               "gsw_sound_speed",
               "gsw_sound_speed_ice",
               "gsw_sound_speed_t_exact",
               "gsw_specvol_anom_standard",
               "gsw_specvol",
               "gsw_specvol_ice",
               "gsw_specvol_sso_0",
               "gsw_specvol_t_exact",
               "gsw_sp_from_c",
               "gsw_sp_from_sa_baltic",
               "gsw_sp_from_sa",
               "gsw_sp_from_sk",
               "gsw_sp_from_sr",
               "gsw_sp_from_sstar",
               "gsw_spiciness0",
               "gsw_spiciness1",
               "gsw_spiciness2",
               "gsw_sr_from_sp",
               "gsw_sstar_from_sa",
               "gsw_sstar_from_sp",
               "gsw_t_deriv_chem_potential_water_t_exact",
               "gsw_t_freezing",
               "gsw_t_freezing_exact",
               "gsw_t_freezing_poly",
               "gsw_t_from_ct",
               "gsw_t_from_pt0_ice",
               "gsw_thermobaric",
               "gsw_util_indx",
               "gsw_util_xinterp1",
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

def handle_nan(f):
    def wrapper(*args):
        if any(isnan(a) for a in args):
            return nan
        else:
            return f(*args)
    return wrapper

def addname(line, doc=None):
    """ Pull a function from the cgsw namespace into the gsw namespace """
    name = line.split(" ", 2)[2].split("(", 1)[0]
    if name[:4] == "gsw_":
        exec("{0} = vectorize(handle_nan(cgsw.{1}), doc=\"{2}\")".format(name[4:], name, doc), addname.__globals__)
    return

for line in lines:
    name = cname(line)
    if name in importnames:
        func = getfuncpointer(name)
        func.argtypes = argtypes(line)
        func.restype = restype(line)
        doc = name + str(argnames(line))
        addname(line, doc=doc)
