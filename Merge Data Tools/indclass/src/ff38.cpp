#include <Rcpp.h>
using namespace Rcpp;

//' Convert SIC codes to Fama French 38 industry codes
//'
//' Converts SIC codes to their corresponding industry code using the Fama-French
//' 38 industry portfolio classifications
//'
//' @param SIC A numeric vector of SIC codes
//' @return A numeric vector of Fama-French 38 industry portfolio codes
//' @export
//' @examples
//' x <- c(800,2000,4537)
//' sic_to_ff38(x)
// [[Rcpp::export]]
IntegerVector sic_to_ff38(IntegerVector SIC) {
  int n = SIC.size();
  IntegerVector out = no_init(n);
  bool warn = false;

  for (int i = 0; i < n; ++i) {
    if (IntegerVector::is_na(SIC[i])) {
      out[i] = NA_REAL;
    } else if ((SIC[i] < 100) || (SIC[i] > 9999)) {
      warn = true;
      out[i] = NA_REAL;
    } else if ((SIC[i] >= 100) && (SIC[i] <= 999)) {
      out[i] = 1;
    } else if ((SIC[i] >= 1000) && (SIC[i] <= 1299)) {
      out[i] = 2;
    } else if ((SIC[i] >= 1300) && (SIC[i] <= 1399)) {
      out[i] = 3;
    } else if ((SIC[i] >= 1400) && (SIC[i] <= 1499)) {
      out[i] = 4;
    } else if ((SIC[i] >= 1500) && (SIC[i] <= 1799)) {
      out[i] = 5;
    } else if ((SIC[i] >= 2000) && (SIC[i] <= 2099)) {
      out[i] = 6;
    } else if ((SIC[i] >= 2100) && (SIC[i] <= 2199)) {
      out[i] = 7;
    } else if ((SIC[i] >= 2200) && (SIC[i] <= 2299)) {
      out[i] = 8;
    } else if ((SIC[i] >= 2300) && (SIC[i] <= 2399)) {
      out[i] = 9;
    } else if ((SIC[i] >= 2400) && (SIC[i] <= 2499)) {
      out[i] = 10;
    } else if ((SIC[i] >= 2500) && (SIC[i] <= 2599)) {
      out[i] = 11;
    } else if ((SIC[i] >= 2600) && (SIC[i] <= 2661)) {
      out[i] = 12;
    } else if ((SIC[i] >= 2700) && (SIC[i] <= 2799)) {
      out[i] = 13;
    } else if ((SIC[i] >= 2800) && (SIC[i] <= 2899)) {
      out[i] = 14;
    } else if ((SIC[i] >= 2900) && (SIC[i] <= 2999)) {
      out[i] = 15;
    } else if ((SIC[i] >= 3000) && (SIC[i] <= 3099)) {
      out[i] = 16;
    } else if ((SIC[i] >= 3100) && (SIC[i] <= 3199)) {
      out[i] = 17;
    } else if ((SIC[i] >= 3200) && (SIC[i] <= 3299)) {
      out[i] = 18;
    } else if ((SIC[i] >= 3300) && (SIC[i] <= 3399)) {
      out[i] = 19;
    } else if ((SIC[i] >= 3400) && (SIC[i] <= 3499)) {
      out[i] = 20;
    } else if ((SIC[i] >= 3500) && (SIC[i] <= 3599)) {
      out[i] = 21;
    } else if ((SIC[i] >= 3600) && (SIC[i] <= 3699)) {
      out[i] = 22;
    } else if ((SIC[i] >= 3700) && (SIC[i] <= 3799)) {
      out[i] = 23;
    } else if ((SIC[i] >= 3800) && (SIC[i] <= 3879)) {
      out[i] = 24;
    } else if ((SIC[i] >= 3900) && (SIC[i] <= 3999)) {
      out[i] = 25;
    } else if ((SIC[i] >= 4000) && (SIC[i] <= 4799)) {
      out[i] = 26;
    } else if ((SIC[i] >= 4800) && (SIC[i] <= 4829)) {
      out[i] = 27;
    } else if ((SIC[i] >= 4830) && (SIC[i] <= 4899)) {
      out[i] = 28;
    } else if ((SIC[i] >= 4900) && (SIC[i] <= 4949)) {
      out[i] = 29;
    } else if ((SIC[i] >= 4950) && (SIC[i] <= 4959)) {
      out[i] = 30;
    } else if ((SIC[i] >= 4960) && (SIC[i] <= 4969)) {
      out[i] = 31;
    } else if ((SIC[i] >= 4970) && (SIC[i] <= 4979)) {
      out[i] = 32;
    } else if ((SIC[i] >= 5000) && (SIC[i] <= 5199)) {
      out[i] = 33;
    } else if ((SIC[i] >= 5200) && (SIC[i] <= 5999)) {
      out[i] = 34;
    } else if ((SIC[i] >= 6000) && (SIC[i] <= 6999)) {
      out[i] = 35;
    } else if ((SIC[i] >= 7000) && (SIC[i] <= 8999)) {
      out[i] = 36;
    } else if ((SIC[i] >= 9000) && (SIC[i] <= 9999)) {
      out[i] = 37;
    } else {
      out[i] = 38;
    }
  }
  if (warn){
    warning("Valid SIC codes should be integers between 100 and 9999.  One or more of the inputs were outside this range.  NA's were returned for these inputs");
  }

  return out;
}

//' Convert Fama-French 38 industry codes to labels
//'
//' Converts a vector of Fama-French industry codes to the appropriate labels
//'
//' @param ff38 A numeric vector of fama-french industry codes
//' @return A vector of Fama-French 38 industry portfolio labels
//' @export
//' @examples
//' x <- c(1,14,37)
//' ff38_label(x)
// [[Rcpp::export]]
CharacterVector ff38_label(IntegerVector ff38) {
  int n = ff38.size();
  CharacterVector out = no_init(n);
  bool warn = false;

  for (int i = 0; i < n; ++i) {
    if (NumericVector::is_na(ff38[i])) {
      out[i] = NA_REAL;
    } else if ((ff38[i] < 1) || (ff38[i] > 38)) {
      warn = true;
      out[i] = NA_REAL;
    } else if (ff38[i] == 1) {
      out[i] = "Agric";
    } else if (ff38[i] == 2) {
      out[i] = "Mines";
    } else if (ff38[i] == 3) {
      out[i] = "Oil";
    } else if (ff38[i] == 4) {
      out[i] = "Stone";
    } else if (ff38[i] == 5) {
      out[i] = "Cnstr";
    } else if (ff38[i] == 6) {
      out[i] = "Food";
    } else if (ff38[i] == 7) {
      out[i] = "Smoke";
    } else if (ff38[i] == 8) {
      out[i] = "Txtls";
    } else if (ff38[i] == 9) {
      out[i] = "Apprl";
    } else if (ff38[i] == 10) {
      out[i] = "Wood";
    } else if (ff38[i] == 11) {
      out[i] = "Chair";
    } else if (ff38[i] == 12) {
      out[i] = "Paper";
    } else if (ff38[i] == 13) {
      out[i] = "Print";
    } else if (ff38[i] == 14) {
      out[i] = "Chems";
    } else if (ff38[i] == 15) {
      out[i] = "Ptrlm";
    } else if (ff38[i] == 16) {
      out[i] = "Rubbr";
    } else if (ff38[i] == 17) {
      out[i] = "Lethr";
    } else if (ff38[i] == 18) {
      out[i] = "Glass";
    } else if (ff38[i] == 19) {
      out[i] = "Metal";
    } else if (ff38[i] == 20) {
      out[i] = "MtlPr";
    } else if (ff38[i] == 21) {
      out[i] = "Machn";
    } else if (ff38[i] == 22) {
      out[i] = "Elctr";
    } else if (ff38[i] == 23) {
      out[i] = "Cars";
    } else if (ff38[i] == 24) {
      out[i] = "Instr";
    } else if (ff38[i] == 25) {
      out[i] = "Manuf";
    } else if (ff38[i] == 26) {
      out[i] = "Trans";
    } else if (ff38[i] == 27) {
      out[i] = "Phone";
    } else if (ff38[i] == 28) {
      out[i] = "TV";
    } else if (ff38[i] == 29) {
      out[i] = "Utils";
    } else if (ff38[i] == 30) {
      out[i] = "Garbg";
    } else if (ff38[i] == 31) {
      out[i] = "Steam";
    } else if (ff38[i] == 32) {
      out[i] = "Water";
    } else if (ff38[i] == 33) {
      out[i] = "Whlsl";
    } else if (ff38[i] == 34) {
      out[i] = "Rtail";
    } else if (ff38[i] == 35) {
      out[i] = "Money";
    } else if (ff38[i] == 36) {
      out[i] = "Srvc";
    } else if (ff38[i] == 37) {
      out[i] = "Govt";
    } else {
      out[i] = "Other";
    }
  }
  if (warn){
    warning("Valid Fama-French 38 industry codes should be integers between 1 and 38.  One or more of the inputs were outside this range.  NA's were returned for these inputs");
  }

  return out;
}

//' Convert Fama-French 38 industry codes to descriptions
//'
//' Converts a vector of Fama-French industry codes to the appropriate descriptions
//'
//' @param ff38 A numeric vector of fama-french industry codes
//' @return A vector of Fama-French 38 industry portfolio descriptions
//' @export
//' @examples
//' x <- c(1,14,37)
//' ff38_desc(x)
// [[Rcpp::export]]
CharacterVector ff38_desc(IntegerVector ff38) {
  int n = ff38.size();
  CharacterVector out = no_init(n);
  bool warn = false;

  for (int i = 0; i < n; ++i) {
    if (NumericVector::is_na(ff38[i])) {
      out[i] = NA_REAL;
    } else if ((ff38[i] < 1) || (ff38[i] > 38)) {
      warn = true;
      out[i] = NA_REAL;
    } else if (ff38[i] == 1) {
      out[i] = "Agriculture, forestry, and fishing";
    } else if (ff38[i] == 2) {
      out[i] = "Mining";
    } else if (ff38[i] == 3) {
      out[i] = "Oil and Gas Extraction";
    } else if (ff38[i] == 4) {
      out[i] = "Nonmetalic Minerals Except Fuels";
    } else if (ff38[i] == 5) {
      out[i] = "Construction";
    } else if (ff38[i] == 6) {
      out[i] = "Food and Kindred Products";
    } else if (ff38[i] == 7) {
      out[i] = "Tobacco Products";
    } else if (ff38[i] == 8) {
      out[i] = "Textile Mill Products";
    } else if (ff38[i] == 9) {
      out[i] = "Apparel and other Textile Products";
    } else if (ff38[i] == 10) {
      out[i] = "Lumber and Wood Products";
    } else if (ff38[i] == 11) {
      out[i] = "Furniture and Fixtures";
    } else if (ff38[i] == 12) {
      out[i] = "Paper and Allied Products";
    } else if (ff38[i] == 13) {
      out[i] = "Printing and Publishing";
    } else if (ff38[i] == 14) {
      out[i] = "Chemicals and Allied Products";
    } else if (ff38[i] == 15) {
      out[i] = "Petroleum and Coal Products";
    } else if (ff38[i] == 16) {
      out[i] = "Rubber and Miscellaneous Plastics Products";
    } else if (ff38[i] == 17) {
      out[i] = "Leather and Leather Products";
    } else if (ff38[i] == 18) {
      out[i] = "Stone, Clay and Glass Products";
    } else if (ff38[i] == 19) {
      out[i] = "Primary Metal Industries";
    } else if (ff38[i] == 20) {
      out[i] = "Fabricated Metal Products";
    } else if (ff38[i] == 21) {
      out[i] = "Machinery, Except Electrical";
    } else if (ff38[i] == 22) {
      out[i] = "Electrical and Electronic Equipment";
    } else if (ff38[i] == 23) {
      out[i] = "Transportation Equipment";
    } else if (ff38[i] == 24) {
      out[i] = "Instruments and Related Products";
    } else if (ff38[i] == 25) {
      out[i] = "Miscellaneous Manufacturing Industries";
    } else if (ff38[i] == 26) {
      out[i] = "Transportation";
    } else if (ff38[i] == 27) {
      out[i] = "Telephone and Telegraph Communication";
    } else if (ff38[i] == 28) {
      out[i] = "Radio and Television Broadcasting";
    } else if (ff38[i] == 29) {
      out[i] = "Electric, Gas, and Water Supply";
    } else if (ff38[i] == 30) {
      out[i] = "Sanitary Services";
    } else if (ff38[i] == 31) {
      out[i] = "Steam Supply";
    } else if (ff38[i] == 32) {
      out[i] = "Irrigation Systems";
    } else if (ff38[i] == 33) {
      out[i] = "Wholesale";
    } else if (ff38[i] == 34) {
      out[i] = "Retail Stores";
    } else if (ff38[i] == 35) {
      out[i] = "Finance, Insurance, and Real Estate";
    } else if (ff38[i] == 36) {
      out[i] = "Services";
    } else if (ff38[i] == 37) {
      out[i] = "Public Administration";
    } else {
      out[i] = "Almost Nothing";
    }
  }
  if (warn){
    warning("Valid Fama-French 38 industry codes should be integers between 1 and 38.  One or more of the inputs were outside this range.  NA's were returned for these inputs");
  }

  return out;
}
