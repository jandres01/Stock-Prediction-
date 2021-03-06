// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// sic_to_ff38
IntegerVector sic_to_ff38(IntegerVector SIC);
RcppExport SEXP _indclass_sic_to_ff38(SEXP SICSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< IntegerVector >::type SIC(SICSEXP);
    rcpp_result_gen = Rcpp::wrap(sic_to_ff38(SIC));
    return rcpp_result_gen;
END_RCPP
}
// ff38_label
CharacterVector ff38_label(IntegerVector ff38);
RcppExport SEXP _indclass_ff38_label(SEXP ff38SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< IntegerVector >::type ff38(ff38SEXP);
    rcpp_result_gen = Rcpp::wrap(ff38_label(ff38));
    return rcpp_result_gen;
END_RCPP
}
// ff38_desc
CharacterVector ff38_desc(IntegerVector ff38);
RcppExport SEXP _indclass_ff38_desc(SEXP ff38SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< IntegerVector >::type ff38(ff38SEXP);
    rcpp_result_gen = Rcpp::wrap(ff38_desc(ff38));
    return rcpp_result_gen;
END_RCPP
}
// naics_group
IntegerVector naics_group(IntegerVector NAICS, int digit);
RcppExport SEXP _indclass_naics_group(SEXP NAICSSEXP, SEXP digitSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< IntegerVector >::type NAICS(NAICSSEXP);
    Rcpp::traits::input_parameter< int >::type digit(digitSEXP);
    rcpp_result_gen = Rcpp::wrap(naics_group(NAICS, digit));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_indclass_sic_to_ff38", (DL_FUNC) &_indclass_sic_to_ff38, 1},
    {"_indclass_ff38_label", (DL_FUNC) &_indclass_ff38_label, 1},
    {"_indclass_ff38_desc", (DL_FUNC) &_indclass_ff38_desc, 1},
    {"_indclass_naics_group", (DL_FUNC) &_indclass_naics_group, 2},
    {NULL, NULL, 0}
};

RcppExport void R_init_indclass(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
