#include <Rcpp.h>
using namespace Rcpp;

//' Convert full NAICS codes to 2, 3, 4, or 5 digit codes
//'
//' Converts NAICS industry classifications into their broader categories by
//' shortening the 6 digit code to 2, 3, 4, or 5 digits
//'
//' @param NAICS A numeric vector of NAICS codes
//' @param digit How many digits to shorten the NAICS code by
//' @return A numeric vector of shortened NAICS industry codes
//' @export
//' @examples
//' x <- c(800000,205000,190000)
//' naics_group(x, 3)
// [[Rcpp::export]]
IntegerVector naics_group(IntegerVector NAICS, int digit) {
  int n = NAICS.size();
  IntegerVector out = no_init(n);

  if ((digit < 2) || (digit > 5)) {
    stop("second argument, 'digit', must be set to an integer value 2 through 5");
  } else if (digit == 2) {
      for (int i = 0; i < n; ++i) {
        if (NAICS[i] < 100) {
          out[i] = NAICS[i];
        } else if (NAICS[i] < 1000) {
          out[i] = NAICS[i] / 10;
        } else if (NAICS[i] < 10000) {
          out[i] = NAICS[i] / 100;
        } else if (NAICS[i] < 100000) {
          out[i] = NAICS[i] / 1000;
        } else {
          out[i] = NAICS[i] / 10000;
        }
      }
  } else if (digit == 3) {
    for (int i = 0; i < n; ++i) {
      if (NAICS[i] < 1000) {
        out[i] = NAICS[i];
      } else if (NAICS[i] < 10000) {
        out[i] = NAICS[i] / 10;
      } else if (NAICS[i] < 100000) {
        out[i] = NAICS[i] / 100;
      } else {
        out[i] = NAICS[i] / 1000;
      }
    }
  } else if (digit == 4) {
    for (int i = 0; i < n; ++i) {
      if (NAICS[i] < 10000) {
        out[i] = NAICS[i];
      } else if (NAICS[i] < 100000) {
        out[i] = NAICS[i] / 10;
      } else {
        out[i] = NAICS[i] / 100;
      }
    }
  } else {
    for (int i = 0; i < n; ++i) {
      if (NAICS[i] < 100000) {
        out[i] = NAICS[i];
      } else {
        out[i] = NAICS[i] / 10;
      }
    }
  }

  return out;
}
