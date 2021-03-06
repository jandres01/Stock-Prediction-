% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{ff38_desc}
\alias{ff38_desc}
\title{Convert Fama-French 38 industry codes to descriptions}
\usage{
ff38_desc(ff38)
}
\arguments{
\item{ff38}{A numeric vector of fama-french industry codes}
}
\value{
A vector of Fama-French 38 industry portfolio descriptions
}
\description{
Converts a vector of Fama-French industry codes to the appropriate descriptions
}
\examples{
x <- c(1,14,37)
ff38_desc(x)
}
